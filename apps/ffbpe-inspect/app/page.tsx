"use client";

import { FFBPE, type BpeEncoder } from "@tokn-ai/ffbpe/browser";
import { inspect, type Inspection } from "@tokn-ai/ffbpe-inspect";
import {
  TOKENIZER_PRESETS,
  getPreset,
  loadPreset,
  type TokenizerPreset,
} from "@tokn-ai/ffbpe-presets";
import { useEffect, useMemo, useState } from "react";

const SPECIAL_TOKEN = "<|endoftext|>";
const DEFAULT_PRESET: TokenizerPreset = "cl100k_base";
const encoder_promises = new Map<TokenizerPreset, Promise<BpeEncoder>>();

const EXAMPLES = [
  {
    label: "Mixed script",
    text: "Tokenizers don't read words — they build them. 你好 👋",
  },
  {
    label: "Whitespace",
    text: "one  two\n\tthree",
  },
  {
    label: "Special token",
    text: `chapter one${SPECIAL_TOKEN}chapter two`,
  },
] as const;

function visibleText(text: string): string {
  if (text === " ") return "space";
  if (text === "\n") return "newline";
  if (text === "\t") return "tab";
  return text.replaceAll("\n", "↵\n").replaceAll("\t", "⇥");
}

function tokenLabel(text: string | null, byte_hex: string): string {
  if (text === null) return `0x${byte_hex.replaceAll(" ", "")}`;
  return visibleText(text);
}

function presetEncoder(name: TokenizerPreset): Promise<BpeEncoder> {
  const cached = encoder_promises.get(name);
  if (cached !== undefined) return cached;
  const loading = loadPreset(name, {
    model_url: new URL(`models/${name}.tiktoken`, location.href),
  }).catch(error => {
    encoder_promises.delete(name);
    throw error;
  });
  encoder_promises.set(name, loading);
  return loading;
}

export default function Home() {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [selected_preset, setSelectedPreset] = useState<TokenizerPreset>(DEFAULT_PRESET);
  const [loaded_encoder, setLoadedEncoder] = useState<{
    name: TokenizerPreset;
    encoder: BpeEncoder;
  } | null>(null);
  const [init_error, setInitError] = useState<string | null>(null);
  const [active_pretoken, setActivePretoken] = useState(0);
  const [active_token, setActiveToken] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    FFBPE.init()
      .then(() => presetEncoder(selected_preset))
      .then(next_encoder => {
        if (!cancelled) {
          setLoadedEncoder({ name: selected_preset, encoder: next_encoder });
        }
      })
      .catch(reason => {
        if (!cancelled) {
          setInitError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selected_preset]);

  const encoder = loaded_encoder?.name === selected_preset
    ? loaded_encoder.encoder
    : null;
  const selected_definition = getPreset(selected_preset);

  const inspection = useMemo<{ result: Inspection | null; error: string | null }>(() => {
    if (encoder === null) return { result: null, error: null };
    try {
      return { result: inspect(encoder, text), error: null };
    } catch (reason) {
      return {
        result: null,
        error: reason instanceof Error ? reason.message : String(reason),
      };
    }
  }, [encoder, text]);
  const { result } = inspection;
  const error = init_error ?? inspection.error;

  const selected_token = active_token === null
    ? null
    : result?.tokens[active_token] ?? null;
  const selected_pretoken = result?.pretokens[active_pretoken] ?? null;

  function selectExample(example_text: string) {
    setText(example_text);
    setActivePretoken(0);
    setActiveToken(null);
  }

  return (
    <main id="top">
      <header className="site-header">
        <div className="site-header-inner">
          <a className="site-brand" href="../" aria-label="FFBPE home">
            <span className="site-brand-mark" aria-hidden="true">
              <b>FF</b><i>/</i><b>BPE</b>
            </span>
            <span className="site-brand-section">INSPECT</span>
          </a>
          <nav className="site-nav" aria-label="Primary navigation">
            <a href="../#why">Why FFBPE</a>
            <a href="../#benchmarks">Benchmarks</a>
            <a href="./" aria-current="page">Inspect</a>
            <a href="../docs/">Docs</a>
            <a href="https://github.com/tokn-ai/ffbpe">GitHub</a>
          </nav>
        </div>
      </header>

      <section className="workbench" aria-label="Tokenizer inspector">
        <div className="input-header">
          <div>
            <span className="step-kicker">INPUT</span>
            <h2>Give it something interesting.</h2>
            <p className="runtime-note">
              <span className="live-dot" aria-hidden="true" />
              WASM · LOCAL MODELS · NO DATA LEAVES THIS TAB
            </p>
          </div>
          <div className="input-controls">
            <label className="preset-picker">
              <span>TOKENIZER PRESET</span>
              <select
                value={selected_preset}
                onChange={event => {
                  setSelectedPreset(event.target.value as TokenizerPreset);
                  setInitError(null);
                  setActivePretoken(0);
                  setActiveToken(null);
                }}
              >
                {TOKENIZER_PRESETS.map(preset => (
                  <option key={preset.name} value={preset.name}>
                    {preset.display_name} · {preset.vocab_size.toLocaleString()} vocab
                  </option>
                ))}
              </select>
              <small>{selected_definition.description}</small>
            </label>
            <div className="examples" aria-label="Text examples">
              {EXAMPLES.map(example => (
                <button
                  className={text === example.text ? "example active" : "example"}
                  key={example.label}
                  onClick={() => selectExample(example.text)}
                  type="button"
                >
                  {example.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <label className="input-wrap">
          <span className="sr-only">Text to tokenize</span>
          <textarea
            value={text}
            onChange={event => {
              setText(event.target.value);
              setActivePretoken(0);
              setActiveToken(null);
            }}
            placeholder="Type or paste text…"
            spellCheck={false}
          />
          <span className="byte-count">{result?.byte_count ?? 0} UTF-8 BYTES</span>
        </label>

        {error !== null ? (
          <p className="error" role="alert">Could not inspect this text: {error}</p>
        ) : result === null ? (
          <div className="loading" role="status">
            <span /> Downloading and verifying {selected_definition.display_name} in your browser…
          </div>
        ) : (
          <>
            <div className="pipeline-heading">
              <span>THE PIPELINE</span>
              <div className="stats">
                <b>{selected_definition.display_name}</b>
                <i />
                <strong>{result.pretoken_count}</strong> PRETOKENS
                <i />
                <strong>{result.token_count}</strong> TOKENS
                <i />
                <strong>{result.token_count === 0 ? "0.00" : (result.byte_count / result.token_count).toFixed(2)}</strong> BYTES / TOKEN
              </div>
            </div>

            <section className="stage pretoken-stage" aria-labelledby="pretoken-title">
              <header className="stage-title">
                <span className="stage-number">01</span>
                <div>
                  <h2 id="pretoken-title">Pretokenizer</h2>
                  <p>Pattern boundaries and special tokens</p>
                </div>
              </header>
              <div className="span-row" aria-label="Pretokens">
                {result.pretokens.length === 0 ? (
                  <p className="empty">No text, no boundaries.</p>
                ) : result.pretokens.map((pretoken, index) => (
                  <button
                    className={`pretoken pretoken-${index % 5}${active_pretoken === index ? " selected" : ""}`}
                    key={`${pretoken.start_byte}-${pretoken.end_byte}`}
                    onClick={() => {
                      setActivePretoken(index);
                      setActiveToken(null);
                    }}
                    type="button"
                  >
                    <span className="pretoken-text">{visibleText(pretoken.text)}</span>
                    <span className="pretoken-meta">
                      {pretoken.kind === "special" ? "SPECIAL" : `B${pretoken.start_byte}–${pretoken.end_byte}`}
                    </span>
                  </button>
                ))}
              </div>
            </section>

            <div className="pipeline-join" aria-hidden="true">
              <span>PRETOKENS ENTER BPE</span>
              <i>↓</i>
            </div>

            <section className="stage token-stage" aria-labelledby="token-title">
              <header className="stage-title">
                <span className="stage-number">02</span>
                <div>
                  <h2 id="token-title">BPE tokenizer</h2>
                  <p>Learned merges become vocabulary IDs</p>
                </div>
              </header>
              <div className="token-groups">
                {result.pretokens.length === 0 ? (
                  <p className="empty">Tokens will appear here.</p>
                ) : result.pretokens.map((pretoken, pretoken_index) => (
                  <div
                    className={`token-group group-${pretoken_index % 5}${active_pretoken === pretoken_index ? " selected" : ""}`}
                    key={`${pretoken.start_byte}-${pretoken.end_byte}`}
                    onMouseEnter={() => setActivePretoken(pretoken_index)}
                  >
                    <span className="group-label">{visibleText(pretoken.text)}</span>
                    <div className="group-tokens">
                      {result.tokens
                        .slice(pretoken.token_start, pretoken.token_end)
                        .map((token, relative_index) => {
                          const token_index = pretoken.token_start + relative_index;
                          return (
                            <button
                              className={active_token === token_index ? "token selected" : "token"}
                              key={`${token.start_byte}-${token.id}`}
                              onClick={() => {
                                setActivePretoken(pretoken_index);
                                setActiveToken(token_index);
                              }}
                              type="button"
                            >
                              <span>{tokenLabel(token.text, token.byte_hex)}</span>
                              <small>#{token.id}</small>
                            </button>
                          );
                        })}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <aside className="detail-bar" aria-live="polite">
              <span className="detail-label">INSPECTED</span>
              {selected_token === null ? (
                <>
                  <strong>{selected_pretoken === null ? "Nothing selected" : visibleText(selected_pretoken.text)}</strong>
                  <span>
                    {selected_pretoken === null
                      ? "Choose a pretoken or BPE token"
                      : `${selected_pretoken.token_end - selected_pretoken.token_start} token${selected_pretoken.token_end - selected_pretoken.token_start === 1 ? "" : "s"} · bytes ${selected_pretoken.start_byte}–${selected_pretoken.end_byte}`}
                  </span>
                </>
              ) : (
                <>
                  <strong>{tokenLabel(selected_token.text, selected_token.byte_hex)}</strong>
                  <span>ID #{selected_token.id} · bytes {selected_token.start_byte}–{selected_token.end_byte} · hex {selected_token.byte_hex}</span>
                </>
              )}
              <span className="detail-hint">CLICK ANY PIECE TO INSPECT</span>
            </aside>
          </>
        )}
      </section>
    </main>
  );
}
