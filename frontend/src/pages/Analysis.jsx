import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  BarChart3,
  Sparkles,
  Tag,
  Smile,
  Meh,
  Frown,
  MapPin,
  Layers,
  Activity,
  AlertCircle,
} from 'lucide-react';

/**
 * Analysis page: visualises the policy-analysis bundle produced by
 *   - GET /api/analysis/overview (when backend is reachable), or
 *   - the static /analysis.json file shipped via Vite's public folder
 *     (fallback for GitHub Pages / static hosting).
 */
const Analysis = () => {
  const [bundle, setBundle] = useState(null);
  const [error, setError] = useState(null);
  const [source, setSource] = useState(null); // "api" | "static"
  const [topicKind, setTopicKind] = useState('lda'); // "lda" | "semantic"
  const [activeTopic, setActiveTopic] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      // 1) Try live API
      try {
        const res = await axios.get('/api/analysis/overview', { timeout: 4000 });
        if (!cancelled && res.data && res.data.summary) {
          setBundle(res.data);
          setSource('api');
          return;
        }
      } catch (_) {
        /* fall through to static */
      }

      // 2) Fall back to static JSON in public/
      try {
        const url = `${import.meta.env.BASE_URL}analysis.json`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) {
          setBundle(data);
          setSource('static');
        }
      } catch (e) {
        if (!cancelled) setError('Could not load analysis data.');
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-20 text-center">
        <AlertCircle className="mx-auto w-12 h-12 text-amber-500 mb-3" />
        <h2 className="text-xl font-semibold text-slate-800">{error}</h2>
        <p className="text-slate-500 mt-2">
          Run <code className="bg-slate-100 px-1.5 py-0.5 rounded">python backend/generate_static_analysis.py</code> to
          generate the analysis bundle, or start the FastAPI backend.
        </p>
      </div>
    );
  }

  if (!bundle) {
    return (
      <div className="flex justify-center py-32">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600" />
      </div>
    );
  }

  const topics = topicKind === 'lda' ? bundle.lda_topics : bundle.semantic_topics;
  const schemesPerTopic =
    topicKind === 'lda' ? bundle.schemes_per_lda_topic : bundle.schemes_per_semantic_topic;
  const safeActive = Math.min(activeTopic, Math.max(0, topics.length - 1));
  const activeTopicSchemes = schemesPerTopic?.[safeActive] || [];

  return (
    <div className="max-w-7xl mx-auto px-4 py-10 space-y-10">
      <Header bundle={bundle} source={source} />

      <SummaryGrid bundle={bundle} />

      <TopicExplorer
        topics={topics}
        topicKind={topicKind}
        setTopicKind={setTopicKind}
        activeTopic={safeActive}
        setActiveTopic={setActiveTopic}
        schemes={activeTopicSchemes}
        semanticBackend={bundle.summary?.semantic_backend}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <SentimentPanel bundle={bundle} />
        <KeywordCloud cloud={bundle.keyword_cloud} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <BarPanel
          title="Schemes per category"
          icon={<Layers className="w-5 h-5" />}
          rows={bundle.category_distribution.map((r) => ({ label: r.category, value: r.count }))}
        />
        <BarPanel
          title="Schemes per state"
          icon={<MapPin className="w-5 h-5" />}
          rows={bundle.state_distribution.slice(0, 12).map((r) => ({ label: r.state, value: r.count }))}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <PolarityPanel
          title="Average sentiment by state"
          rows={bundle.state_sentiment.slice(0, 12)}
          labelKey="state"
        />
        <PolarityPanel
          title="Average sentiment by category"
          rows={bundle.category_sentiment}
          labelKey="category"
        />
      </div>
    </div>
  );
};

// ─────────────── sub-components ───────────────

const Header = ({ bundle, source }) => (
  <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200 pb-6">
    <div>
      <div className="flex items-center gap-2 text-primary-600 font-medium">
        <BarChart3 className="w-5 h-5" />
        <span>Policy analysis dashboard</span>
      </div>
      <h1 className="text-3xl font-bold text-slate-900 mt-1">
        NLP insights across {bundle.summary?.n_schemes || 0} government schemes
      </h1>
      <p className="text-slate-600 mt-2 max-w-3xl">
        Topic modeling (LDA), semantic clustering, keyword extraction and lexicon-based
        sentiment analysis, computed over the active scheme corpus.
      </p>
    </div>
    <div className="text-xs text-slate-500">
      Source:{' '}
      <span
        className={`inline-block px-2 py-0.5 rounded-full font-medium ${
          source === 'api' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'
        }`}
      >
        {source === 'api' ? 'live backend' : 'static bundle'}
      </span>
    </div>
  </div>
);

const SummaryGrid = ({ bundle }) => {
  const s = bundle.sentiment_summary || {};
  const cards = [
    { label: 'Schemes analysed', value: bundle.summary?.n_schemes ?? 0, icon: <Activity className="w-5 h-5" />, tone: 'slate' },
    { label: 'LDA topics', value: bundle.summary?.n_topics_lda ?? 0, icon: <Sparkles className="w-5 h-5" />, tone: 'indigo' },
    { label: 'Semantic clusters', value: bundle.summary?.n_topics_semantic ?? 0, icon: <Layers className="w-5 h-5" />, tone: 'violet' },
    { label: 'Avg sentiment polarity', value: (s.avg_polarity ?? 0).toFixed(2), icon: <Smile className="w-5 h-5" />, tone: 'emerald' },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div key={c.label} className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <div className={`text-${c.tone}-600 mb-2`}>{c.icon}</div>
          <div className="text-2xl font-bold text-slate-900">{c.value}</div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mt-1">{c.label}</div>
        </div>
      ))}
    </div>
  );
};

const TopicExplorer = ({ topics, topicKind, setTopicKind, activeTopic, setActiveTopic, schemes, semanticBackend }) => (
  <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
    <div className="flex flex-col md:flex-row md:items-center justify-between mb-5 gap-3">
      <div>
        <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary-600" />
          Topic explorer
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Discovered automatically from scheme descriptions. Click any topic to see its schemes.
        </p>
      </div>
      <div className="inline-flex bg-slate-100 rounded-lg p-1">
        <button
          onClick={() => {
            setTopicKind('lda');
            setActiveTopic(0);
          }}
          className={`px-3 py-1.5 text-sm rounded-md transition ${
            topicKind === 'lda' ? 'bg-white shadow text-slate-900' : 'text-slate-600'
          }`}
        >
          LDA
        </button>
        <button
          onClick={() => {
            setTopicKind('semantic');
            setActiveTopic(0);
          }}
          className={`px-3 py-1.5 text-sm rounded-md transition ${
            topicKind === 'semantic' ? 'bg-white shadow text-slate-900' : 'text-slate-600'
          }`}
        >
          Semantic{semanticBackend === 'sbert' ? ' (BERT)' : ''}
        </button>
      </div>
    </div>

    {topics.length === 0 ? (
      <p className="text-slate-500 text-sm">No topics available.</p>
    ) : (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-2">
          {topics.map((t, i) => (
            <button
              key={t.topic_id}
              onClick={() => setActiveTopic(i)}
              className={`w-full text-left rounded-xl border px-4 py-3 transition ${
                i === activeTopic
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-900">{t.label}</span>
                <span className="text-xs text-slate-500">{t.scheme_count} schemes</span>
              </div>
              <div className="text-xs text-slate-500 mt-1 truncate">
                {t.top_words.slice(0, 5).join(' · ')}
              </div>
            </button>
          ))}
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              Top words for "{topics[activeTopic]?.label}"
            </div>
            <div className="flex flex-wrap gap-2">
              {(topics[activeTopic]?.top_words || []).map((w) => (
                <span
                  key={w}
                  className="inline-block bg-white border border-slate-200 px-3 py-1 rounded-full text-sm text-slate-700"
                >
                  {w}
                </span>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wide text-slate-500">Schemes in this topic</div>
            {schemes.length === 0 ? (
              <p className="text-sm text-slate-500">No schemes mapped to this topic.</p>
            ) : (
              schemes.map((s) => (
                <div
                  key={s.scheme_id}
                  className="flex items-start justify-between gap-4 bg-white rounded-lg border border-slate-200 px-4 py-3"
                >
                  <div className="flex-1">
                    <div className="font-medium text-slate-900">{s.name}</div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      {s.category} · {s.state}
                    </div>
                    {s.keywords?.length > 0 && (
                      <div className="text-xs text-slate-500 mt-1.5">
                        Keywords: {s.keywords.join(', ')}
                      </div>
                    )}
                  </div>
                  <SentimentBadge label={s.sentiment} />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    )}
  </section>
);

const SentimentBadge = ({ label }) => {
  const conf = {
    positive: { bg: 'bg-emerald-100', text: 'text-emerald-700', icon: <Smile className="w-3.5 h-3.5" /> },
    neutral: { bg: 'bg-slate-100', text: 'text-slate-600', icon: <Meh className="w-3.5 h-3.5" /> },
    negative: { bg: 'bg-rose-100', text: 'text-rose-700', icon: <Frown className="w-3.5 h-3.5" /> },
  }[label] || { bg: 'bg-slate-100', text: 'text-slate-600', icon: <Meh className="w-3.5 h-3.5" /> };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${conf.bg} ${conf.text}`}>
      {conf.icon}
      {label || 'neutral'}
    </span>
  );
};

const SentimentPanel = ({ bundle }) => {
  const s = bundle.sentiment_summary || { positive: 0, neutral: 0, negative: 0, total: 0 };
  const total = Math.max(1, s.total || s.positive + s.neutral + s.negative);
  const pct = (n) => Math.round((n * 100) / total);
  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
        <Smile className="w-5 h-5 text-emerald-500" />
        Policy tone
      </h2>
      <p className="text-sm text-slate-500 mt-1">
        Lexicon-based sentiment over scheme descriptions. Polarity ∈ [-1, +1].
      </p>
      <div className="mt-5 space-y-3">
        <ToneRow label="Positive" count={s.positive} pct={pct(s.positive)} color="bg-emerald-500" />
        <ToneRow label="Neutral" count={s.neutral} pct={pct(s.neutral)} color="bg-slate-400" />
        <ToneRow label="Negative" count={s.negative} pct={pct(s.negative)} color="bg-rose-500" />
      </div>
      <div className="mt-5 pt-5 border-t border-slate-200 text-sm text-slate-600">
        Average polarity:{' '}
        <span className="font-semibold text-slate-900">
          {(s.avg_polarity ?? 0).toFixed(3)}
        </span>
      </div>
    </section>
  );
};

const ToneRow = ({ label, count, pct, color }) => (
  <div>
    <div className="flex justify-between text-sm">
      <span className="text-slate-700">{label}</span>
      <span className="text-slate-500">
        {count} · {pct}%
      </span>
    </div>
    <div className="h-2 bg-slate-100 rounded-full mt-1 overflow-hidden">
      <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  </div>
);

const KeywordCloud = ({ cloud }) => {
  const max = Math.max(1, ...cloud.map((c) => c.count));
  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
        <Tag className="w-5 h-5 text-primary-600" />
        Keyword cloud
      </h2>
      <p className="text-sm text-slate-500 mt-1">Top TF-IDF terms across the corpus.</p>
      <div className="mt-4 flex flex-wrap gap-2">
        {cloud.slice(0, 60).map((c) => {
          const ratio = c.count / max;
          const fontSize = 11 + Math.round(ratio * 10);
          return (
            <span
              key={c.word}
              style={{ fontSize: `${fontSize}px` }}
              className="bg-slate-100 text-slate-700 rounded-full px-3 py-1"
            >
              {c.word}
              <span className="ml-1 text-slate-400 text-xs">({c.count})</span>
            </span>
          );
        })}
      </div>
    </section>
  );
};

const BarPanel = ({ title, icon, rows }) => {
  const max = Math.max(1, ...rows.map((r) => r.value));
  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
        {icon} {title}
      </h2>
      <div className="mt-4 space-y-2">
        {rows.map((r) => (
          <div key={r.label}>
            <div className="flex justify-between text-sm">
              <span className="text-slate-700 capitalize">{r.label}</span>
              <span className="text-slate-500">{r.value}</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full mt-1 overflow-hidden">
              <div className="h-full bg-primary-500" style={{ width: `${(r.value / max) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

const PolarityPanel = ({ title, rows, labelKey }) => {
  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
        <Activity className="w-5 h-5 text-primary-600" />
        {title}
      </h2>
      <div className="mt-4 divide-y divide-slate-100">
        {rows.map((r) => {
          const pct = Math.round(((r.avg_polarity + 1) / 2) * 100); // map [-1,+1] → [0,100]
          const color =
            r.avg_polarity > 0.1 ? 'bg-emerald-500' : r.avg_polarity < -0.1 ? 'bg-rose-500' : 'bg-slate-400';
          return (
            <div key={r[labelKey]} className="py-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-700 capitalize">
                  {r[labelKey]} <span className="text-slate-400">({r.schemes})</span>
                </span>
                <span className="text-slate-600 font-medium">{r.avg_polarity.toFixed(2)}</span>
              </div>
              <div className="h-2 bg-slate-100 rounded-full mt-1 overflow-hidden relative">
                <div className="absolute top-0 left-1/2 -translate-x-1/2 h-full w-px bg-slate-300" />
                <div
                  className={`h-full ${color}`}
                  style={{ width: `${Math.abs(r.avg_polarity) * 50}%`, marginLeft: r.avg_polarity >= 0 ? '50%' : `${50 - Math.abs(r.avg_polarity) * 50}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default Analysis;
