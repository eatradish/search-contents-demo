use std::{
    collections::HashSet, env::args, fs, io::{BufRead, BufReader, Read, Seek}, path::{Path, PathBuf}, sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    }, thread
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use flate2::bufread::GzDecoder;
use lzzzz::lz4f::BufReadDecompressor;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::debug;
use winnow::{
    combinator::{separated, separated_pair},
    error::ParserError,
    token::{take_till, take_until},
    PResult, Parser,
};
use zstd::Decoder;

const ZSTD_MAGIC: &[u8] = &[40, 181, 47, 253];
const LZ4_MAGIC: &[u8] = &[0x04, 0x22, 0x4d, 0x18];
const GZIP_MAGIC: &[u8] = &[0x1F, 0x8B];
const BIN_PREFIX: &str = "usr/bin";

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Provides,
    Files,
    BinProvides,
    BinFiles,
}

impl Mode {
    fn paths(&self, dir: &Path) -> Result<Vec<PathBuf>, OmaContentsError> {
        use std::fs;

        let contains_name = "_Contents-";

        let mut paths = vec![];

        for i in fs::read_dir(dir)
            .map_err(|e| OmaContentsError::FailedToOperateDirOrFile(dir.display().to_string(), e))?
            .flatten()
        {
            if i.file_name()
                .into_string()
                .is_ok_and(|x| x.contains(contains_name))
            {
                paths.push(i.path());
            }
        }

        Ok(paths)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OmaContentsError {
    #[error("Contents does not exist")]
    ContentsNotExist,
    #[error("Execute ripgrep failed: {0:?}")]
    ExecuteRgFailed(std::io::Error),
    #[error("Failed to read dir or file: {0}, kind: {1}")]
    FailedToOperateDirOrFile(String, std::io::Error),
    #[error("Failed to get file {0} metadata: {1}")]
    FailedToGetFileMetadata(String, std::io::Error),
    #[error("Failed to wait ripgrep to exit: {0}")]
    FailedToWaitExit(std::io::Error),
    #[error("Failed to build Aho-Corasick tree: {0}")]
    FailedToBuildAhoCorasick(#[from] aho_corasick::BuildError),
    #[error("Contents entry missing path list: {0}")]
    ContentsEntryMissingPathList(String),
    #[error("Command not found wrong argument")]
    CnfWrongArgument,
    #[error("Ripgrep exited with error")]
    RgWithError,
    #[error(transparent)]
    LzzzErr(#[from] lzzzz::lz4f::Error),
    #[error("")]
    NoResult,
    #[error("Illegal file: {0}")]
    IllegalFile(String),
}

pub fn pure_search(
    path: impl AsRef<Path>,
    mode: Mode,
    query: &str,
    cb: impl Fn(usize) + Sync + Send + 'static,
) -> Result<Vec<(String, String)>, OmaContentsError> {
    let paths = mode.paths(path.as_ref())?;
    let count = Arc::new(AtomicUsize::new(0));
    let count_local = count.clone();

    let searcher = AhoCorasickBuilder::new().build(&[query])?;
    let query = query.to_string();

    let worker = thread::spawn(move || {
        paths
            .par_iter()
            .map(
                move |path| -> Result<Vec<(String, String)>, OmaContentsError> {
                    pure_search_contents_from_path(path, &searcher, &query, &count, mode)
                },
            )
            .collect::<Result<Vec<_>, OmaContentsError>>()
            .map(|x| x.into_iter().flatten().collect::<Vec<_>>())
    });

    let mut tmp = 0;
    loop {
        let count = count_local.load(Ordering::Acquire);
        if count > 0 && count != tmp {
            cb(count);
            tmp = count;
        }

        if worker.is_finished() {
            return worker.join().unwrap();
        }
    }
}

fn pure_search_contents_from_path(
    path: &Path,
    searcher: &AhoCorasick,
    query: &str,
    count: &AtomicUsize,
    mode: Mode,
) -> Result<Vec<(String, String)>, OmaContentsError> {
    let mut f = fs::File::open(path)
        .map_err(|e| OmaContentsError::FailedToOperateDirOrFile(path.display().to_string(), e))?;

    let mut buf = [0; 4];
    f.read_exact(&mut buf).ok();
    f.rewind().map_err(|e| {
        debug!("{e}");
        OmaContentsError::IllegalFile(path.display().to_string())
    })?;

    let ext = path.extension().and_then(|x| x.to_str());

    let contents_reader: &mut dyn Read = match ext {
        Some("zst") => {
            check_file_magic_4bytes(buf, path, ZSTD_MAGIC)?;
            // https://github.com/gyscos/zstd-rs/issues/281
            &mut Decoder::new(BufReader::new(f)).unwrap()
        }
        Some("lz4") => {
            check_file_magic_4bytes(buf, path, LZ4_MAGIC)?;
            &mut BufReadDecompressor::new(BufReader::new(f))?
        }
        Some("gz") => {
            if buf[..2] != *GZIP_MAGIC {
                return Err(OmaContentsError::IllegalFile(path.display().to_string()));
            }
            &mut GzDecoder::new(BufReader::new(f))
        }
        _ => &mut BufReader::new(f),
    };

    let reader = BufReader::new(contents_reader);

    let cb: Box<dyn Fn(&str, &str, &str) -> bool> = match mode {
        Mode::Provides => Box::new(|_pkg: &str, file: &str, _: &str| searcher.is_match(file)),
        Mode::Files => Box::new(|pkg: &str, _file: &str, query: &str| pkg == query),
        Mode::BinProvides => Box::new(|_pkg: &str, file: &str, _: &str| {
            file.starts_with(BIN_PREFIX) && searcher.is_match(file)
        }),
        Mode::BinFiles => {
            Box::new(|pkg: &str, file: &str, query: &str| pkg == query && file.starts_with(BIN_PREFIX))
        }
    };

    let res = pure_search_foreach_result(cb, reader, count, query);

    Ok(res)
}

fn pure_search_foreach_result(
    cb: impl Fn(&str, &str, &str) -> bool,
    reader: BufReader<&mut dyn Read>,
    count: &AtomicUsize,
    query: &str,
) -> Vec<(String, String)> {
    let mut res = HashSet::new();

    for i in reader.lines() {
        let i = match i {
            Ok(i) => i,
            Err(_) => continue,
        };

        let (file, pkgs) = match single_line::<()>(&mut i.as_str()) {
            Ok(line) => line,
            Err(_) => continue,
        };

        for (_, pkg) in pkgs {
            if cb(pkg, file, query) {
                count.fetch_add(1, Ordering::AcqRel);
                let line = (pkg.to_string(), prefix(file));
                if !res.contains(&line) {
                    res.insert(line);
                }
            }
        }
    }

    res.into_iter().collect()
}

#[inline]
fn prefix(s: &str) -> String {
    if s.starts_with('/') {
        s.to_string()
    } else {
        "/".to_owned() + s
    }
}

#[inline]
fn check_file_magic_4bytes(
    buf: [u8; 4],
    path: &Path,
    magic: &[u8],
) -> Result<(), OmaContentsError> {
    if buf != magic {
        return Err(OmaContentsError::IllegalFile(path.display().to_string()));
    }

    Ok(())
}

#[inline]
fn pkg_split<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<(&'a str, &'a str), E> {
    separated_pair(take_till(0.., '/'), pkg_name_sep, second_single).parse_next(input)
}

#[inline]
fn pkg_name_sep<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<(), E> {
    "/".void().parse_next(input)
}

#[inline]
fn second_single<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<&'a str, E> {
    take_till(0.., |c| c == ',' || c == '\n').parse_next(input)
}

#[inline]
fn second<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<Vec<(&'a str, &'a str)>, E> {
    separated(0.., pkg_split, ',').parse_next(input)
}

#[inline]
fn first<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<&'a str, E> {
    take_until(1.., "   ").parse_next(input)
}

#[inline]
fn sep<'a, E: ParserError<&'a str>>(input: &mut &'a str) -> PResult<(), E> {
    "   ".void().parse_next(input)
}

type ContentsLine<'a> = (&'a str, Vec<(&'a str, &'a str)>);

#[inline]
pub fn single_line<'a, E: ParserError<&'a str>>(
    input: &mut &'a str,
) -> PResult<ContentsLine<'a>, E> {
    separated_pair(first, sep, second).parse_next(input)
}

pub type ContentsLines<'a> = Vec<(&'a str, Vec<(&'a str, &'a str)>)>;

#[inline]
pub fn parse_contents<'a, E: ParserError<&'a str>>(
    input: &mut &'a str,
) -> PResult<ContentsLines<'a>, E> {
    use winnow::combinator::{repeat, terminated};
    repeat(1.., terminated(single_line, "\n")).parse_next(input)
}

fn main() {
    let mut args = args().skip(1);
    let path = args.next().unwrap();
    let query = args.next().unwrap();
    dbg!(pure_search(&path, Mode::Provides, &query, |_| {}).unwrap());
}
