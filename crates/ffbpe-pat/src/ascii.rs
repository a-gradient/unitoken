//! Portable ASCII run scanning, adapted from GigaToken's SWAR approach.
//! See LICENSE-GIGATOKEN and README.md for attribution.

const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
const CASE_BITS: u64 = 0x2020_2020_2020_2020;
const REPEAT_BYTE: u64 = 0x0101_0101_0101_0101;

#[inline]
pub(super) fn scan_letters(bytes: &[u8], start: usize) -> usize {
  scan_range::<b'a', b'z', true>(bytes, start)
}

#[inline]
pub(super) fn scan_lowercase(bytes: &[u8], start: usize) -> usize {
  scan_range::<b'a', b'z', false>(bytes, start)
}

#[inline]
pub(super) fn scan_uppercase(bytes: &[u8], start: usize) -> usize {
  scan_range::<b'A', b'Z', false>(bytes, start)
}

#[inline]
fn scan_range<const LOW: u8, const HIGH: u8, const FOLD: bool>(
  bytes: &[u8],
  mut pos: usize,
) -> usize {
  while bytes.len() - pos >= 8 {
    // A checked, little-endian load keeps this independent of alignment and
    // host endianness. No padding or reads beyond the input are required.
    let word = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
    if word & HIGH_BITS != 0 {
      break;
    }
    let folded = if FOLD { word | CASE_BITS } else { word };
    // Every lane is ASCII. Setting the high bits before subtraction prevents
    // borrowing between lanes, so each high bit is an independent comparison.
    let above_low = (folded | HIGH_BITS).wrapping_sub(u64::from(LOW) * REPEAT_BYTE);
    let below_high = ((u64::from(HIGH) * REPEAT_BYTE) | HIGH_BITS).wrapping_sub(folded);
    let nonmatching = !(above_low & below_high) & HIGH_BITS;
    if nonmatching != 0 {
      return pos + nonmatching.trailing_zeros() as usize / 8;
    }
    pos += 8;
  }
  while let Some(&byte) = bytes.get(pos) {
    let folded = if FOLD { byte | 0x20 } else { byte };
    if folded.wrapping_sub(LOW) > HIGH - LOW {
      break;
    }
    pos += 1;
  }
  pos
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn runs_match_scalar_for_all_bytes_alignments_and_tails() {
    type Scanner = fn(&[u8], usize) -> usize;
    type Predicate = fn(&u8) -> bool;
    for (scan, predicate, fill) in [
      (
        scan_letters as Scanner,
        u8::is_ascii_alphabetic as Predicate,
        b'a',
      ),
      (scan_lowercase, u8::is_ascii_lowercase, b'a'),
      (scan_uppercase, u8::is_ascii_uppercase, b'Z'),
    ] {
      for start in 0..16 {
        for length in 0..=33 {
          for byte in 0..=255 {
            let mut bytes = vec![fill; start + length];
            for pos in start..bytes.len() {
              bytes[pos] = byte;
              let expected = start + bytes[start..].iter().take_while(|b| predicate(b)).count();
              assert_eq!(
                scan(&bytes, start),
                expected,
                "start={start}, pos={pos}, byte={byte}"
              );
              bytes[pos] = fill;
            }
            assert_eq!(scan(&bytes, start), bytes.len());
          }
        }
      }
    }
  }
}
