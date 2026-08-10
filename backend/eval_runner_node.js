const fs = require('fs');
const path = require('path');
const http = require('http');

const API_URL = 'http://127.0.0.1:8000/api/chat';
const BENCHMARK_PATH = path.join(__dirname, 'eval', 'benchmark_dataset.json');
const OUTPUT_PATH = path.join(__dirname, 'eval', 'baseline_results.json');

function postQuestion(question, sessionId) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      question: question,
      session_id: sessionId
    });

    const req = http.request(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      },
      timeout: 300000
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          resolve({ status: res.statusCode, data: json });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data });
        }
      });
    });

    req.on('error', err => reject(err));
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout after 300s'));
    });

    req.write(payload);
    req.end();
  });
}

function computeNGramOverlap(text1, text2, n = 5) {
  const words1 = (text1 || '').split(/\s+/).map(w => w.toLowerCase()).filter(w => w.length > 2);
  const words2 = (text2 || '').split(/\s+/).map(w => w.toLowerCase()).filter(w => w.length > 2);

  if (words1.length < n || words2.length < n) return 0.0;

  const ngrams1 = new Set();
  for (let i = 0; i <= words1.length - n; i++) {
    ngrams1.add(words1.slice(i, i + n).join(' '));
  }
  const ngrams2 = new Set();
  for (let i = 0; i <= words2.length - n; i++) {
    ngrams2.add(words2.slice(i, i + n).join(' '));
  }

  if (ngrams1.size === 0) return 0.0;

  let intersection = 0;
  for (let item of ngrams1) {
    if (ngrams2.has(item)) intersection++;
  }
  return intersection / ngrams1.size;
}

async function runBenchmark() {
  console.log('================================================================================');
  console.log('            NODE.JS RAG BENCHMARK & EVALUATION RUNNER                           ');
  console.log('================================================================================');

  if (!fs.existsSync(BENCHMARK_PATH)) {
    console.error(`Benchmark dataset not found at ${BENCHMARK_PATH}`);
    process.exit(1);
  }

  const benchmark = JSON.parse(fs.readFileSync(BENCHMARK_PATH, 'utf-8'));
  console.log(`Loaded ${benchmark.length} test cases.\n`);

  const results = [];
  const sessionId = '00000000-0000-0000-0000-000000000000';

  let totalLatency = 0;
  let totalGrounding = 0;
  let totalCorrectness = 0;
  let totalCitation = 0;
  let totalCompleteness = 0;
  let passedCount = 0;
  let failedCount = 0;
  let reasoningLeakageCount = 0;
  let verbatimCopyCount = 0;

  for (let idx = 0; idx < benchmark.length; idx++) {
    const testCase = benchmark[idx];
    const category = testCase.category || 'general';
    const question = testCase.question;
    const expectedAnswer = testCase.expected_answer;
    const reqKeywords = testCase.required_keywords || [];
    const forbidKeywords = testCase.forbidden_keywords || [];
    const expDocs = testCase.expected_documents || [];

    console.log(`[${idx + 1}/${benchmark.length}] [${category.toUpperCase()}] Q: ${question.slice(0, 60)}...`);

    const startTime = Date.now();
    let actualAnswer = '';
    let citations = [];
    let processingTimeMs = 0;
    let statusCode = 0;
    let rawJson = null;

    try {
      const res = await postQuestion(question, sessionId);
      statusCode = res.status;
      const duration = Date.now() - startTime;

      if (statusCode === 200) {
        rawJson = res.data;
        actualAnswer = rawJson.answer || '';
        citations = rawJson.citations || [];
        processingTimeMs = rawJson.processing_time_ms || duration;
      } else {
        actualAnswer = `HTTP ${statusCode}: ${JSON.stringify(res.data || res.raw)}`;
        processingTimeMs = duration;
      }
    } catch (err) {
      statusCode = 500;
      actualAnswer = `Error: ${err.message}`;
      processingTimeMs = Date.now() - startTime;
    }

    totalLatency += processingTimeMs;
    const lowerAns = actualAnswer.toLowerCase();

    // 1. Reasoning leakage
    let hasLeakage = false;
    if (lowerAns.includes('<think>') || lowerAns.includes('</think>') || lowerAns.includes('let me analyze') || lowerAns.includes('looking at passage')) {
      hasLeakage = true;
      reasoningLeakageCount++;
    }
    for (let kw of forbidKeywords) {
      if (lowerAns.includes(kw.toLowerCase())) {
        hasLeakage = true;
      }
    }

    // 2. Verbatim copy
    let verbatimOverlap = 0.0;
    for (let cite of citations) {
      const snippet = cite.preview || cite.chunk_text || '';
      if (snippet) {
        const overlap = computeNGramOverlap(actualAnswer, snippet, 5);
        if (overlap > verbatimOverlap) verbatimOverlap = overlap;
      }
    }
    const isVerbatim = verbatimOverlap > 0.40;
    if (isVerbatim) verbatimCopyCount++;

    // 3. Grounding
    let groundingScore = 100.0;
    if (category === 'negative questions') {
      groundingScore = lowerAns.includes('information not found') ? 100.0 : 0.0;
    } else {
      groundingScore = (citations.length > 0 || !lowerAns.includes('information not found')) ? 100.0 : 50.0;
    }

    // 4. Keyword Completeness
    const matchedKeywords = reqKeywords.filter(kw => lowerAns.includes(kw.toLowerCase()));
    const completenessRatio = reqKeywords.length > 0 ? (matchedKeywords.length / reqKeywords.length) : 1.0;
    const completenessScore = Math.round(completenessRatio * 1000) / 10;

    // 5. Citation Accuracy
    let citationScore = 100.0;
    if (expDocs.length > 0) {
      const citedDocs = citations.map(c => c.document_title || '');
      const docMatches = expDocs.filter(doc => citedDocs.some(cd => cd.toLowerCase().includes(doc.toLowerCase())));
      citationScore = Math.round((docMatches.length / expDocs.length) * 1000) / 10;
    } else {
      citationScore = citations.length === 0 ? 100.0 : 80.0;
    }

    // 6. Overall Correctness
    let correctnessScore = 100.0;
    let failureDomain = 'none';
    const issues = [];

    if (category === 'negative questions' && !lowerAns.includes('information not found')) {
      correctnessScore -= 50.0;
      issues.push('Failed negative question refusal constraint');
      failureDomain = 'prompt';
    }
    if (hasLeakage) {
      correctnessScore -= 30.0;
      issues.push('Reasoning leakage detected');
      failureDomain = 'sanitization';
    }
    if (isVerbatim) {
      correctnessScore -= 20.0;
      issues.push(`Verbatim copying threshold exceeded (${(verbatimOverlap * 100).toFixed(1)}%)`);
      failureDomain = 'prompt';
    }
    if (completenessRatio < 0.5) {
      correctnessScore -= 30.0;
      issues.push(`Low keyword completeness (${matchedKeywords.length}/${reqKeywords.length})`);
      if (failureDomain === 'none') failureDomain = 'LLM';
    }
    if (expDocs.length > 0 && citationScore < 50.0) {
      correctnessScore -= 20.0;
      issues.push('Expected citations missing');
      if (failureDomain === 'none') failureDomain = 'retrieval';
    }

    correctnessScore = Math.max(0.0, correctnessScore);
    const passed = correctnessScore >= 70.0;
    if (passed) passedCount++; else failedCount++;

    totalGrounding += groundingScore;
    totalCorrectness += correctnessScore;
    totalCitation += citationScore;
    totalCompleteness += completenessScore;

    results.push({
      id: testCase.id || (idx + 1),
      category: category,
      question: question,
      expected_answer: expectedAnswer,
      actual_answer: actualAnswer,
      citations: citations,
      latency_ms: processingTimeMs,
      correctness_score: correctnessScore,
      grounding_score: groundingScore,
      citation_score: citationScore,
      completeness_score: completenessScore,
      has_reasoning_leakage: hasLeakage,
      is_verbatim_copy: isVerbatim,
      passed: passed,
      issues: issues,
      failure_domain: failureDomain
    });

    console.log(`   -> Status: ${statusCode} | Score: ${correctnessScore} | Latency: ${processingTimeMs}ms | Passed: ${passed}`);
  }

  const summary = {
    total_tests: benchmark.length,
    passed_count: passedCount,
    failed_count: failedCount,
    accuracy_pct: Math.round((passedCount / benchmark.length) * 1000) / 10,
    avg_correctness: Math.round((totalCorrectness / benchmark.length) * 10) / 10,
    avg_grounding: Math.round((totalGrounding / benchmark.length) * 10) / 10,
    avg_citation_accuracy: Math.round((totalCitation / benchmark.length) * 10) / 10,
    avg_completeness: Math.round((totalCompleteness / benchmark.length) * 10) / 10,
    avg_latency_ms: Math.round((totalLatency / benchmark.length) * 10) / 10,
    reasoning_leakage_count: reasoningLeakageCount,
    verbatim_copy_count: verbatimCopyCount,
    results: results
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(summary, null, 2), 'utf-8');
  console.log('\n================================================================================');
  console.log(`BENCHMARK COMPLETE: ${passedCount}/${benchmark.length} Passed (${summary.accuracy_pct}%)`);
  console.log(`Avg Correctness: ${summary.avg_correctness}% | Grounding: ${summary.avg_grounding}% | Citation: ${summary.avg_citation_accuracy}%`);
  console.log(`Avg Latency: ${summary.avg_latency_ms}ms | Baseline saved to ${OUTPUT_PATH}`);
  console.log('================================================================================');
}

runBenchmark().catch(err => {
  console.error('Fatal benchmark error:', err);
  process.exit(1);
});
