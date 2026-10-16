//! Turning a file's bytes into a document's title and text, by file type.
//!
//! Everything here runs in Rust, so syncing works before the Python
//! trainer is installed. A file that can't be read well is skipped with a
//! reason instead of being stored as garbage.

use std::io::{Cursor, Read};
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::Event;

/// File extensions synk can read.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["md", "markdown", "txt", "html", "htm", "docx", "pdf"];

/// A PDF page with fewer extracted characters than this on average is
/// probably a scan without a text layer.
const MIN_PDF_CHARS_PER_PAGE: usize = 50;
/// More than this share of replacement or control characters means the
/// PDF's font maps are broken and the text would be gibberish.
const MAX_GARBLED_SHARE: f64 = 0.02;

pub fn is_supported(path: &Path) -> bool {
    format(path).is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Markdown,
    Text,
    Html,
    Docx,
    Pdf,
}

fn format(path: &Path) -> Option<Format> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "md" | "markdown" => Format::Markdown,
        "txt" => Format::Text,
        "html" | "htm" => Format::Html,
        "docx" => Format::Docx,
        "pdf" => Format::Pdf,
        _ => return None,
    })
}

/// Why a file was skipped, in words a person understands. Stable values:
/// agents may branch on them.
pub mod reason {
    pub const NOT_UTF8: &str = "not_utf8";
    pub const UNREADABLE: &str = "unreadable";
    pub const NO_TEXT_LAYER: &str = "no_text_layer";
    pub const GARBLED_TEXT: &str = "garbled_text";
    pub const EMPTY: &str = "empty";
}

/// Changes when the text synk gets from the same file changes, so the next
/// sync reads every file again. 2: Unicode NFC.
pub const READER_VERSION: u32 = 2;

/// Unicode NFC: one code point for "ệ" rather than "e" plus two combining
/// marks, however the file was typed or exported. The same Vietnamese word
/// is then one word for chunking, caching and the model.
pub fn nfc(text: &str) -> String {
    use unicode_normalization::{UnicodeNormalization, is_nfc_quick};
    match is_nfc_quick(text.chars()) {
        unicode_normalization::IsNormalized::Yes => text.to_owned(),
        _ => text.nfc().collect(),
    }
}

/// `(title, normalized text)`, or a skip reason.
pub fn read(path: &Path, bytes: &[u8]) -> Result<(String, String), &'static str> {
    let stem = || {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let (title, text) = match format(path).ok_or(reason::UNREADABLE)? {
        Format::Markdown => {
            let raw = std::str::from_utf8(bytes).map_err(|_| reason::NOT_UTF8)?;
            let text = normalize(strip_front_matter(raw));
            (first_heading(&text), text)
        }
        Format::Text => {
            let raw = std::str::from_utf8(bytes).map_err(|_| reason::NOT_UTF8)?;
            (None, normalize(raw))
        }
        Format::Html => {
            let raw = String::from_utf8_lossy(bytes);
            let markdown = htmd::HtmlToMarkdown::builder()
                .skip_tags(vec![
                    "head", "script", "style", "noscript", "template", "svg",
                ])
                .build()
                .convert(&raw)
                .map_err(|_| reason::UNREADABLE)?;
            let text = normalize(&unescape_markdown(&markdown));
            (html_title(&raw).or_else(|| first_heading(&text)), text)
        }
        Format::Docx => {
            let (title, text) = docx(bytes).ok_or(reason::UNREADABLE)?;
            let text = normalize(&text);
            (title.or_else(|| first_heading(&text)), text)
        }
        Format::Pdf => {
            let text = pdf(bytes)?;
            (None, text)
        }
    };
    if text.is_empty() {
        return Err(reason::EMPTY);
    }
    let title = title.filter(|t| !t.is_empty()).unwrap_or_else(stem);
    Ok((nfc(&title), nfc(&text)))
}

fn strip_front_matter(raw: &str) -> &str {
    let Some(rest) = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
    else {
        return raw;
    };
    match rest.find("\n---") {
        Some(end) => rest[end + 4..]
            .split_once('\n')
            .map_or("", |(_, body)| body),
        None => raw,
    }
}

/// Trims trailing whitespace and collapses runs of blank lines.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0;
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_owned()
}

fn first_heading(text: &str) -> Option<String> {
    text.lines()
        .find_map(|line| line.strip_prefix("# "))
        .map(|title| title.trim().to_owned())
        .filter(|title| !title.is_empty())
}

/// Drops the backslashes the converter puts before punctuation
/// (`keepalive\_timeout`), which would otherwise be embedded as text.
fn unescape_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek().is_some_and(char::is_ascii_punctuation) {
            continue;
        }
        out.push(c);
    }
    out
}

/// The text of the first `<title>` element.
fn html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open_end = start + lower[start..].find('>')? + 1;
    let close = open_end + lower[open_end..].find("</title")?;
    let title = html[open_end..close]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!title.is_empty()).then_some(title)
}

/// Paragraph text from `word/document.xml`, with headings as Markdown
/// headings, and the title from `docProps/core.xml`.
fn docx(bytes: &[u8]) -> Option<(Option<String>, String)> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).ok()?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .ok()?
        .read_to_string(&mut xml)
        .ok()?;
    let mut core = String::new();
    if let Ok(mut file) = archive.by_name("docProps/core.xml") {
        let _ = file.read_to_string(&mut core);
    }

    let mut reader = Reader::from_str(&xml);
    let mut out = String::new();
    let mut paragraph = String::new();
    let mut heading: Option<usize> = None;
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.local_name().into_inner() == "t" => in_text = true,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => match e.local_name().into_inner() {
                "tab" => paragraph.push('\t'),
                "br" | "cr" => paragraph.push('\n'),
                "pStyle" => {
                    let style = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.local_name().into_inner() == "val")
                        .map(|a| a.value.to_ascii_lowercase());
                    heading = style.as_deref().and_then(heading_level);
                }
                _ => {}
            },
            Ok(Event::Text(t)) if in_text => {
                paragraph.push_str(&t);
            }
            Ok(Event::GeneralRef(r)) if in_text => {
                if let Ok(Some(c)) = r.resolve_char_ref() {
                    paragraph.push(c);
                } else {
                    let name: &str = &r;
                    paragraph.push_str(match name {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => "",
                    });
                }
            }
            Ok(Event::End(e)) => match e.local_name().into_inner() {
                "t" => in_text = false,
                "p" => {
                    let text = paragraph.trim();
                    if !text.is_empty() {
                        if let Some(level) = heading {
                            out.push_str(&"#".repeat(level));
                            out.push(' ');
                        }
                        out.push_str(text);
                    }
                    out.push_str("\n\n");
                    paragraph.clear();
                    heading = None;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
    }
    Some((core_title(&core), out))
}

/// `Heading1` / `heading 2` / `Title` -> a Markdown heading level.
fn heading_level(style: &str) -> Option<usize> {
    if style == "title" {
        return Some(1);
    }
    let rest = style.strip_prefix("heading")?.trim();
    rest.parse::<usize>().ok().filter(|l| (1..=6).contains(l))
}

fn core_title(core: &str) -> Option<String> {
    let start = core.find("<dc:title>")? + "<dc:title>".len();
    let end = start + core[start..].find("</dc:title>")?;
    let title = core[start..end].trim();
    (!title.is_empty()).then(|| title.to_owned())
}

/// The text layer of a PDF, page by page, or why it can't be used.
fn pdf(bytes: &[u8]) -> Result<String, &'static str> {
    // pdf-extract panics on some malformed files; a bad PDF must not take
    // the whole sync down, nor print a panic message.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let pages = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes));
    std::panic::set_hook(hook);
    let pages = match pages {
        Ok(Ok(pages)) => pages,
        _ => return Err(reason::UNREADABLE),
    };
    let text = normalize(
        &pages
            .iter()
            .map(|p| normalize(p))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    pdf_quality(&text, pages.len())?;
    Ok(text)
}

/// Rejects text that is almost empty for its page count (a scan) or full
/// of replacement and control characters (broken font maps).
fn pdf_quality(text: &str, pages: usize) -> Result<(), &'static str> {
    let chars = text.chars().filter(|c| !c.is_whitespace()).count();
    if chars < MIN_PDF_CHARS_PER_PAGE * pages.max(1) {
        return Err(reason::NO_TEXT_LAYER);
    }
    let bad = text
        .chars()
        .filter(|c| *c == '\u{FFFD}' || (c.is_control() && !c.is_whitespace()))
        .count();
    if bad as f64 > MAX_GARBLED_SHARE * chars as f64 {
        return Err(reason::GARBLED_TEXT);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn docx_bytes(document: &str, core: Option<&str>) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            zip.start_file("word/document.xml", opts).unwrap();
            zip.write_all(document.as_bytes()).unwrap();
            if let Some(core) = core {
                zip.start_file("docProps/core.xml", opts).unwrap();
                zip.write_all(core.as_bytes()).unwrap();
            }
            zip.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn markdown_front_matter_is_stripped_and_text_normalized() {
        let raw = "---\ntitle: x\n---\n# Heading  \n\n\n\nBody line   \r\n";
        let (title, text) = read(Path::new("a.md"), raw.as_bytes()).unwrap();
        assert_eq!(title, "Heading");
        assert_eq!(text, "# Heading\n\nBody line");
    }

    #[test]
    fn html_becomes_markdown_with_its_title() {
        let html = "<html><head><title>Cấu hình  nginx</title><script>x()</script></head>\
                    <body><h1>Tăng timeout</h1><p>Đặt <b>keepalive_timeout</b> lên 75s.</p></body></html>";
        let (title, text) = read(Path::new("a.html"), html.as_bytes()).unwrap();
        assert_eq!(title, "Cấu hình nginx");
        assert!(text.contains("# Tăng timeout"), "{text}");
        assert!(text.contains("**keepalive_timeout**"), "{text}");
        assert!(!text.contains("x()"), "{text}");
    }

    #[test]
    fn docx_paragraphs_headings_and_title() {
        let doc = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Sao lưu</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Chạy </w:t></w:r><w:r><w:t>pg_dump &amp; nén</w:t></w:r></w:p>
<w:p><w:r><w:t>hằng đêm</w:t></w:r><w:r><w:tab/><w:t>2h</w:t></w:r></w:p>
</w:body></w:document>"#;
        let core = "<cp:coreProperties><dc:title>Quy trình sao lưu</dc:title></cp:coreProperties>";
        let (title, text) = read(Path::new("a.docx"), &docx_bytes(doc, Some(core))).unwrap();
        assert_eq!(title, "Quy trình sao lưu");
        assert_eq!(text, "# Sao lưu\n\nChạy pg_dump & nén\n\nhằng đêm\t2h");
        let (title, _) = read(Path::new("b.docx"), &docx_bytes(doc, None)).unwrap();
        assert_eq!(title, "Sao lưu");
    }

    #[test]
    fn broken_files_are_skipped_with_a_reason() {
        assert_eq!(
            read(Path::new("a.docx"), b"not a zip").unwrap_err(),
            reason::UNREADABLE
        );
        assert_eq!(
            read(Path::new("a.pdf"), b"%PDF-1.4 garbage").unwrap_err(),
            reason::UNREADABLE
        );
        assert_eq!(
            read(Path::new("a.txt"), &[0xff, 0xfe, 0x00]).unwrap_err(),
            reason::NOT_UTF8
        );
        assert_eq!(read(Path::new("a.md"), b"  \n").unwrap_err(), reason::EMPTY);
    }

    /// A one-page PDF with `lines` of Helvetica text, xref offsets computed.
    fn pdf_bytes(lines: &[&str]) -> Vec<u8> {
        let mut content = String::from("BT /F1 12 Tf 72 720 Td 14 TL\n");
        for line in lines {
            content.push_str(&format!("({line}) Tj T*\n"));
        }
        content.push_str("ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_owned(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_owned(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn pdf_text_layer_is_read_and_scans_are_skipped() {
        let pdf = pdf_bytes(&[
            "Backups run every night at two in the morning.",
            "Restore drills happen on the first Monday of each month.",
        ]);
        let (title, text) = read(Path::new("ops/backups.pdf"), &pdf).unwrap();
        assert_eq!(title, "backups");
        assert!(text.contains("Restore drills happen"), "{text}");

        let scan = pdf_bytes(&[]);
        assert_eq!(
            read(Path::new("scan.pdf"), &scan).unwrap_err(),
            reason::NO_TEXT_LAYER
        );
    }

    #[test]
    fn pdf_quality_gate() {
        assert_eq!(pdf_quality("short", 2), Err(reason::NO_TEXT_LAYER));
        let good = "Tăng keepalive_timeout lên 75 giây cho nginx. ".repeat(4);
        assert_eq!(pdf_quality(&good, 1), Ok(()));
        let garbled = format!("{good}{}", "\u{FFFD}".repeat(10));
        assert_eq!(pdf_quality(&garbled, 1), Err(reason::GARBLED_TEXT));
    }
}
