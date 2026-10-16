//! Splitting documents into chunks.
//!
//! Chunks follow paragraph boundaries (blank lines) and are packed up to a
//! maximum size, measured by a length function: the base model's tokens, or
//! characters. A paragraph longer than the maximum is split at line, then
//! sentence, then character boundaries.

/// Splits `text` into chunks of at most `max` units of `len`.
pub fn chunk_by(text: &str, max: usize, len: &dyn Fn(&str) -> usize) -> Vec<String> {
    let max = max.max(1);
    let text = text.replace("\r\n", "\n");
    let mut pieces = Vec::new();
    for block in text.split("\n\n").map(str::trim).filter(|b| !b.is_empty()) {
        split_long(block, max, len, &mut pieces);
    }
    let mut chunks = Vec::new();
    pack(
        pieces.iter().map(String::as_str),
        "\n\n",
        max,
        len,
        &mut chunks,
    );
    chunks
}

/// Splits `text` into chunks of at most `max_chars` characters.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    chunk_by(text, max_chars, &chars)
}

pub fn chars(s: &str) -> usize {
    s.chars().count()
}

fn split_long(block: &str, max: usize, len: &dyn Fn(&str) -> usize, out: &mut Vec<String>) {
    if len(block) <= max {
        out.push(block.to_owned());
        return;
    }
    let lines: Vec<&str> = block.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() > 1 {
        pack(lines.into_iter(), "\n", max, len, out);
        return;
    }
    let sentences = split_sentences(block);
    if sentences.len() > 1 {
        pack(sentences.into_iter(), " ", max, len, out);
        return;
    }
    hard_split(block, max, len, out);
}

/// Packs `parts` into strings of at most `max`, splitting any part that is
/// still too long on its own.
fn pack<'a>(
    parts: impl Iterator<Item = &'a str>,
    sep: &str,
    max: usize,
    len: &dyn Fn(&str) -> usize,
    out: &mut Vec<String>,
) {
    let mut current = String::new();
    for part in parts {
        let part = part.trim();
        if len(part) > max {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            split_long(part, max, len, out);
            continue;
        }
        if !current.is_empty() && len(&format!("{current}{sep}{part}")) > max {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str(sep);
        }
        current.push_str(part);
    }
    if !current.is_empty() {
        out.push(current);
    }
}

/// Cuts text with no usable boundaries into the longest prefixes that fit.
fn hard_split(text: &str, max: usize, len: &dyn Fn(&str) -> usize, out: &mut Vec<String>) {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let mut start = 0; // index into `bounds`
    while start + 1 < bounds.len() {
        // The largest end that fits, but always at least one character.
        let (mut lo, mut hi) = (start + 1, bounds.len() - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if len(&text[bounds[start]..bounds[mid]]) <= max {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let piece = text[bounds[start]..bounds[lo]].trim();
        if !piece.is_empty() {
            out.push(piece.to_owned());
        }
        start = lo;
    }
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let at_boundary = matches!(c, '.' | '!' | '?' | '。' | '！' | '？')
            && chars.peek().is_none_or(|(_, next)| next.is_whitespace());
        if at_boundary {
            let end = i + c.len_utf8();
            out.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_paragraphs_are_packed_together() {
        let text = format!(
            "{}\n\n{}\n\n{}",
            "a".repeat(100),
            "b".repeat(100),
            "c".repeat(100)
        );
        let chunks = chunk_text(&text, 250);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].starts_with('a') && chunks[0].ends_with('b'));
    }

    #[test]
    fn no_chunk_exceeds_the_limit_and_nothing_is_lost() {
        let sentence = "Deploy lên server bị lỗi timeout, phải tăng keepalive. ";
        let text = sentence.repeat(60);
        let chunks = chunk_text(&text, 300);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.chars().count() <= 300));
        let joined: String = chunks.join(" ");
        assert_eq!(
            joined.split_whitespace().count(),
            text.split_whitespace().count()
        );
    }

    #[test]
    fn text_without_breaks_is_split_by_characters() {
        let chunks = chunk_text(&"x".repeat(1000), 400);
        assert_eq!(
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            [400, 400, 200]
        );
    }

    #[test]
    fn any_length_function_works() {
        let words = |s: &str| s.split_whitespace().count();
        let text = "one two three four five. six seven eight nine ten.\n\neleven twelve";
        let chunks = chunk_by(text, 5, &words);
        assert_eq!(
            chunks,
            [
                "one two three four five.",
                "six seven eight nine ten.",
                "eleven twelve"
            ]
        );
        assert!(chunk_by("averyveryverylongword", 1, &words).len() == 1);
    }
}
