import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { DuckDBInstance } from "@duckdb/node-api";

import { createServer } from "../server.js";

async function seedDatabase(databasePath) {
  const instance = await DuckDBInstance.create(databasePath);
  const connection = await instance.connect();
  await connection.run(`
    CREATE TABLE IF NOT EXISTS theme_discovery_run (
      run_id TEXT,
      run_name TEXT,
      date_start DATE,
      date_end DATE,
      frame_minutes INTEGER,
      config_id TEXT,
      config_json TEXT,
      code_commit TEXT,
      data_version TEXT,
      status TEXT,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS graph_snapshot (
      snapshot_id TEXT,
      run_id TEXT,
      trade_date DATE,
      timestamp TIMESTAMP,
      frame_minutes INTEGER,
      market_session TEXT,
      graph_status TEXT,
      available_minutes_since_open INTEGER,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS consensus_theme_candidate (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      trade_date DATE,
      timestamp TIMESTAMP,
      theme_path_id TEXT,
      members_json TEXT,
      member_count INTEGER,
      source_layers_json TEXT,
      consensus_score DOUBLE,
      structure_score DOUBLE,
      cross_layer_consensus_score DOUBLE,
      flow_support_score DOUBLE,
      dtw_flow_support_score DOUBLE,
      volume_support_score DOUBLE,
      large_trade_support_score DOUBLE,
      stability_score DOUBLE,
      semantic_coherence_score DOUBLE,
      theme_quality_score DOUBLE,
      theme_quality_breakdown_json TEXT,
      keep_status TEXT,
      reject_reason TEXT
    );
    CREATE TABLE IF NOT EXISTS theme_semantic_label (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      label_short TEXT,
      label_long TEXT,
      sector_summary TEXT,
      industry_summary TEXT,
      bucket_tags_json TEXT,
      top_companies_json TEXT,
      semantic_coherence_score DOUBLE,
      explanation TEXT,
      semantic_method TEXT,
      semantic_metadata_json TEXT,
      semantic_prompt_text TEXT,
      dictionary_version TEXT,
      created_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS theme_path_lifecycle (
      theme_path_id TEXT,
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      timestamp TIMESTAMP,
      event_type TEXT,
      age_frames INTEGER,
      duration_minutes INTEGER,
      match_score DOUBLE,
      previous_theme_instance_id TEXT,
      member_retention DOUBLE,
      status TEXT,
      transition_parent_path_id TEXT,
      transition_child_path_id TEXT,
      transition_kind TEXT
    );
    CREATE TABLE IF NOT EXISTS theme_membership (
      theme_instance_id TEXT,
      run_id TEXT,
      snapshot_id TEXT,
      theme_path_id TEXT,
      trade_date DATE,
      symbol TEXT,
      member_rank INTEGER,
      contribution_score DOUBLE,
      return_contribution DOUBLE,
      flow_contribution DOUBLE,
      dtw_flow_contribution DOUBLE,
      large_trade_contribution DOUBLE
    );
    CREATE TABLE IF NOT EXISTS frontend_snapshot_cache (
      snapshot_cache_id TEXT,
      snapshot_id TEXT,
      run_id TEXT,
      timestamp TIMESTAMP,
      cache_type TEXT,
      payload_json TEXT,
      payload_version TEXT,
      created_at TIMESTAMP
    );
  `);
  await connection.run(`
    INSERT INTO theme_discovery_run VALUES
    ('run_001', 'Run One', '2026-01-02', '2026-01-02', 5, 'config_001', '{}', 'abc123', 'data_v1', 'completed', NOW());
    INSERT INTO graph_snapshot VALUES
    ('snapshot_001', 'run_001', '2026-01-02', '2026-01-02 14:45:00+00', 5, 'regular', 'complete', 15, NOW());
    INSERT INTO consensus_theme_candidate VALUES
    ('theme_001', 'run_001', 'snapshot_001', '2026-01-02', '2026-01-02 14:45:00+00', 'path_001', '["AAA","BBB"]', 2, '["return_corr_graph","flow_alignment_graph"]', 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, '{"version":"v1"}', 'keep', '');
    INSERT INTO theme_semantic_label VALUES
    ('theme_001', 'run_001', 'snapshot_001', 'Theme: AAA, BBB', 'Theme: AAA, BBB', 'Tech', 'Software', '[]', '["AAA","BBB"]', 0.8, 'Dictionary label', 'dictionary_v1', '{"source":"test"}', '', 'builtin-v1', NOW());
    INSERT INTO theme_path_lifecycle VALUES
    ('path_001', 'theme_001', 'run_001', 'snapshot_001', '2026-01-02 14:45:00+00', 'birth', 1, 5, 1.0, NULL, 1.0, 'active', NULL, NULL, NULL);
    INSERT INTO theme_membership VALUES
    ('theme_001', 'run_001', 'snapshot_001', 'path_001', '2026-01-02', 'AAA', 1, 1.0, 0.0, 0.0, 0.0, 0.0),
    ('theme_001', 'run_001', 'snapshot_001', 'path_001', '2026-01-02', 'BBB', 2, 1.0, 0.0, 0.0, 0.0, 0.0);
    INSERT INTO frontend_snapshot_cache VALUES
    ('cache_001', 'snapshot_001', 'run_001', '2026-01-02 14:45:00+00', 'snapshot_summary', '{"snapshot_id":"snapshot_001","themes":[{"theme_instance_id":"theme_001"}]}', 'v1', NOW());
  `);
  connection.closeSync();
  instance.closeSync();
}

async function withServer(databasePath, run) {
  process.env.STOCKNETV2_DB = databasePath;
  const server = createServer();
  await new Promise((resolve) => server.listen(0, resolve));
  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  try {
    await run(baseUrl);
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) reject(error);
        else resolve();
      });
    });
    delete process.env.STOCKNETV2_DB;
  }
}

test("server exposes read-only T1 run and snapshot routes", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-node-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  await seedDatabase(databasePath);

  await withServer(databasePath, async (baseUrl) => {
    const runsRes = await fetch(`${baseUrl}/api/runs`);
    assert.equal(runsRes.status, 200);
    const runsPayload = await runsRes.json();
    assert.equal(runsPayload.runs[0].run_id, "run_001");

    const timelineRes = await fetch(`${baseUrl}/api/runs/run_001/timeline`);
    assert.equal(timelineRes.status, 200);
    const timelinePayload = await timelineRes.json();
    assert.equal(timelinePayload.snapshots[0].snapshot_id, "snapshot_001");

    const snapshotRes = await fetch(`${baseUrl}/api/snapshots/snapshot_001`);
    assert.equal(snapshotRes.status, 200);
    const snapshotPayload = await snapshotRes.json();
    assert.equal(snapshotPayload.snapshot_id, "snapshot_001");
    assert.equal(snapshotPayload.cache_type, "snapshot_summary");

    const themeRes = await fetch(`${baseUrl}/api/themes/theme_001`);
    assert.equal(themeRes.status, 200);
    const themePayload = await themeRes.json();
    assert.equal(themePayload.theme_instance_id, "theme_001");
    assert.equal(themePayload.semantic.label_short, "Theme: AAA, BBB");
    assert.equal(themePayload.lifecycle.event_type, "birth");
  });
});

test("server exposes progress api and progress page", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-progress-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  const progressPath = path.join(tempDir, "progress.json");
  const logPath = path.join(tempDir, "run.log");
  const liveProgressDir = path.join(tempDir, "windows", "2025-02", "_live_progress");
  await seedDatabase(databasePath);
  fs.mkdirSync(liveProgressDir, { recursive: true });
  fs.writeFileSync(
    progressPath,
    JSON.stringify(
      {
        status: "running",
        run_label: "2025 Q1 qualification",
        total_windows: 3,
        completed_windows: 1,
        total_trade_dates: 62,
        completed_trade_dates: 20,
        current_window_id: "2025-02",
        current_stage: "graph_build",
        dtw_backend: "torch_cuda",
        gpu_name: "NVIDIA GeForce RTX 5090",
        windows: [
          {
            window_id: "2025-02",
            status: "running",
            total_trade_dates: 19,
            completed_trade_dates: 0,
            trade_dates: [],
          },
        ],
      },
      null,
      2,
    ),
  );
  fs.writeFileSync(
    path.join(liveProgressDir, "2025-02-03.json"),
    JSON.stringify(
      {
        trade_date: "2025-02-03",
        status: "running",
        snapshot_id: "qualification_2025-02-03_1435",
        snapshot_index: 1,
        total_snapshots: 78,
        snapshot_clock_code: "1435",
        available_minutes_since_open: 5,
        progress_percent: 1.2821,
      },
      null,
      2,
    ),
  );
  fs.writeFileSync(logPath, "window 2025-01 completed\nwindow 2025-02 started\n");

  process.env.STOCKNETV2_PROGRESS_FILE = progressPath;
  process.env.STOCKNETV2_LOG_FILE = logPath;
  await withServer(databasePath, async (baseUrl) => {
    const progressRes = await fetch(`${baseUrl}/api/progress`);
    assert.equal(progressRes.status, 200);
    const progressPayload = await progressRes.json();
    assert.equal(progressPayload.progress.status, "running");
    assert.equal(progressPayload.progress.completed_windows, 1);
    assert.equal(progressPayload.progress.current_trade_date, "2025-02-03");
    assert.equal(progressPayload.progress.current_snapshot_clock_code, "1435");
    assert.equal(progressPayload.progress.windows[0].trade_dates[0].snapshot_index, 1);
    assert.equal(progressPayload.progress.windows[0].trade_dates[0].elapsed_display.includes(":"), true);
    assert.equal(progressPayload.progress.windows[0].trade_dates[0].avg_snapshot_display, "-");
    assert.match(progressPayload.logs.join("\n"), /\[live\] 2025-02 2025-02-03/);
    assert.match(progressPayload.logs.join("\n"), /window 2025-02 started/);

    const pageRes = await fetch(`${baseUrl}/progress`);
    assert.equal(pageRes.status, 200);
    const pageHtml = await pageRes.text();
    assert.match(pageHtml, /StockNetV2 Qualification Progress/);
    assert.match(pageHtml, /EventSource\("\/api\/progress\/stream"\)/);
    assert.match(pageHtml, /Current Trade Date/);
    assert.match(pageHtml, /Current Snapshot/);
    assert.match(pageHtml, /DTW Backend/);
    assert.match(pageHtml, /GPU/);
    assert.match(pageHtml, /avg_snapshot=/);
  });
  delete process.env.STOCKNETV2_PROGRESS_FILE;
  delete process.env.STOCKNETV2_LOG_FILE;
});

test("server keeps progress stream alive when progress files are not created yet", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-progress-empty-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  await seedDatabase(databasePath);

  process.env.STOCKNETV2_PROGRESS_FILE = path.join(tempDir, "nested", "progress.json");
  process.env.STOCKNETV2_LOG_FILE = path.join(tempDir, "nested", "run.log");

  await withServer(databasePath, async (baseUrl) => {
    await new Promise((resolve, reject) => {
      const req = http.get(`${baseUrl}/api/progress/stream`, (res) => {
        assert.equal(res.statusCode, 200);
        let buffer = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          buffer += chunk;
          if (buffer.includes("data:")) {
            req.destroy();
            resolve();
          }
        });
        res.on("error", reject);
      });
      req.on("error", reject);
    });
  });

  delete process.env.STOCKNETV2_PROGRESS_FILE;
  delete process.env.STOCKNETV2_LOG_FILE;
});

test("server exposes lightweight month progress api and page", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-month-progress-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  await seedDatabase(databasePath);

  fs.writeFileSync(
    path.join(tempDir, "run_config.json"),
    JSON.stringify(
      {
        run_name: "month_run_demo",
        profile: "cpu_full",
        resume_mode: "off",
        date_start: "2026-01-02",
        date_end: "2026-01-03",
        planned_snapshots: 100,
        pending_snapshots: 88,
        trade_dates: [
          { trade_date: "2026-01-02", month: "2026-01", planned_snapshots: 78 },
          { trade_date: "2026-01-03", month: "2026-01", planned_snapshots: 22 },
        ],
      },
      null,
      2,
    ),
  );
  fs.mkdirSync(path.join(tempDir, "month=2026-01", "dates", "date=2026-01-02", "snapshots", "snapshot=1435"), { recursive: true });
  fs.writeFileSync(path.join(tempDir, "month=2026-01", "dates", "date=2026-01-02", "snapshots", "snapshot=1435", "_PROFILE_SUCCESS"), "");
  fs.writeFileSync(
    path.join(tempDir, "progress.jsonl"),
    [
      JSON.stringify({
        status: "snapshot_started",
        trade_date: "2026-01-02",
        snapshot_id: "2026-01-02T14:35:00+00:00",
        snapshot_clock: "1435",
        updated_at: "2026-06-23T07:00:00+00:00",
      }),
      JSON.stringify({
        status: "snapshot_complete",
        trade_date: "2026-01-02",
        snapshot_id: "2026-01-02T14:35:00+00:00",
        snapshot_clock: "1435",
        worker_pid: 1234,
        updated_at: "2026-06-23T07:00:05+00:00",
      }),
    ].join("\n"),
  );
  fs.writeFileSync(
    path.join(tempDir, "run.log"),
    JSON.stringify({
      status: "snapshot_complete",
      trade_date: "2026-01-02",
      snapshot_id: "2026-01-02T14:35:00+00:00",
      snapshot_clock: "1435",
      edge_count: 321,
      worker_pid: 1234,
      updated_at: "2026-06-23T07:00:05+00:00",
    }) + "\n",
  );

  process.env.STOCKNETV2_MONTH_RUN_ROOT = tempDir;
  await withServer(databasePath, async (baseUrl) => {
    const progressRes = await fetch(`${baseUrl}/api/month-progress`);
    assert.equal(progressRes.status, 200);
    const payload = await progressRes.json();
    assert.equal(payload.run_name, "month_run_demo");
    assert.equal(payload.completed_snapshots, 1);
    assert.equal(payload.current_snapshot_clock, "1435");
    assert.equal(payload.current_edge_count, 321);
    assert.equal(payload.trade_date_groups.running[0].trade_date, "2026-01-02");
    assert.equal(payload.trade_date_groups.completed.length, 0);
    assert.equal(payload.trade_date_groups.pending[0].trade_date, "2026-01-03");

    const pageRes = await fetch(`${baseUrl}/month-progress`);
    assert.equal(pageRes.status, 200);
    const pageHtml = await pageRes.text();
    assert.match(pageHtml, /StockNetV2 Month Run Progress/);
    assert.match(pageHtml, /EventSource\("\/api\/month-progress\/stream"\)/);
    assert.match(pageHtml, /处理中/);
    assert.match(pageHtml, /已完成/);
  });
  delete process.env.STOCKNETV2_MONTH_RUN_ROOT;
});

test("server exposes benchmark progress api and renders benchmark details on progress page", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "stocknetv2-benchmark-"));
  const databasePath = path.join(tempDir, "stocknetv2.duckdb");
  const benchmarkStatusPath = path.join(tempDir, "benchmark_status.json");
  const benchmarkLogPath = path.join(tempDir, "benchmark.log");
  await seedDatabase(databasePath);

  fs.writeFileSync(
    benchmarkStatusPath,
    JSON.stringify(
      {
        status: "running",
        current_benchmark_type: "inner_throughput",
        current_candidate_id: "gpu_batch1024_layer2",
        current_trade_date: "2025-01-02",
        current_snapshot_clock_code: "1450",
        decision_state: "gathering_inner_throughput",
        completed_candidates: 1,
        total_candidates: 9,
      },
      null,
      2,
    ),
  );
  fs.writeFileSync(benchmarkLogPath, "inner candidate gpu_batch1024_layer2 started\n");

  process.env.STOCKNETV2_BENCHMARK_STATUS_FILE = benchmarkStatusPath;
  process.env.STOCKNETV2_BENCHMARK_LOG_FILE = benchmarkLogPath;

  await withServer(databasePath, async (baseUrl) => {
    const benchmarkRes = await fetch(`${baseUrl}/api/benchmark-progress`);
    assert.equal(benchmarkRes.status, 200);
    const benchmarkPayload = await benchmarkRes.json();
    assert.equal(benchmarkPayload.benchmark.status, "running");
    assert.equal(benchmarkPayload.benchmark.current_candidate_id, "gpu_batch1024_layer2");
    assert.match(benchmarkPayload.logs.join("\n"), /started/);

    const pageRes = await fetch(`${baseUrl}/progress`);
    assert.equal(pageRes.status, 200);
    const pageHtml = await pageRes.text();
    assert.match(pageHtml, /Benchmark Status/);
    assert.match(pageHtml, /Current Candidate/);
    assert.match(pageHtml, /Decision State/);
  });

  delete process.env.STOCKNETV2_BENCHMARK_STATUS_FILE;
  delete process.env.STOCKNETV2_BENCHMARK_LOG_FILE;
});
