import { useState, useRef, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import SectionHeader from "@/components/SectionHeader";
import InsightBox from "@/components/InsightBox";
import { financialData } from "@/data/mockData";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/components/ui/sonner";

const COLORS = {
  primary: "#58a6ff", green: "#3fb950", red: "#f85149", yellow: "#d29922",
  purple: "#bc8cff", orange: "#ff7b72", border: "#30363d", muted: "#8b949e", text: "#e6edf3",
};

const tooltipStyle = { background: "#21262d", border: `1px solid ${COLORS.border}`, borderRadius: 8, color: COLORS.text, fontFamily: "IBM Plex Mono" };

const COLUMNS_8 = ["Date", "ROE", "ROA", "ROI", "D/E", "Current Ratio", "Quick Ratio", "P/E", "Price/Sales"];
const COLUMNS_14 = [...COLUMNS_8, "Stock Price", "CPI", "Oil Price", "Copper Price", "GDP Growth", "Fed Funds Rate", "Sentiment"];
const TARGET_COLS = ["FHI"];

const API_URL = "https://murtazamajid-apples-fhi-forecasting-analysis.hf.space/predict";

// Map UI column names to API feature names
const COL_TO_API: Record<string, string> = {
  "ROE": "Return on Equity",
  "ROA": "Return on Assets",
  "ROI": "Return on Investment",
  "D/E": "Debt to Equity Ratio",
  "Current Ratio": "Current Ratio",
  "Quick Ratio": "Quick Ratio",
  "P/E": "PE Ratio",
  "Price/Sales": "Price to Sales Ratio",
  "CPI": "CPIAUCSL",
  "Oil Price": "WTISPLC",
  "Copper Price": "PCOPPUSDM",
  "GDP Growth": "GDP",
  "Fed Funds Rate": "FEDFUNDS",
  "Sentiment": "sentiment_score",
};

const models = [
  { id: "arima_ratios", name: "ARIMAX — Ratios Only", type: "arimax", features: "Ratios Only (8)", apiType: "arima_ratios" },
  { id: "arima_all", name: "ARIMAX — All Features", type: "arimax", features: "All Features (14)", apiType: "arima_all" },
  { id: "lstm_ratios", name: "LSTM — Ratios Only", type: "lstm", features: "Ratios Only (8)", apiType: "lstm_ratios" },
  { id: "lstm_all", name: "LSTM — All Features", type: "lstm", features: "All Features (14)", apiType: "lstm_all" },
];

type DataRow = Record<string, string | number>;

function parseCSV(text: string): DataRow[] {
  const lines = text.trim().split("\n").filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const row: DataRow = {};
    headers.forEach((h, i) => {
      const num = parseFloat(vals[i]);
      row[h] = isNaN(num) ? vals[i] : num;
    });
    return row;
  });
}

function defaultRows(): DataRow[] {
  return financialData.slice(-13).map((d) => ({
    Date: d.date.slice(0, 7),
    ROE: +(d.returnOnEquity / 200).toFixed(4),
    ROA: +(d.returnOnAssets / 40).toFixed(4),
    ROI: +(d.returnOnInvestment / 80).toFixed(4),
    "D/E": +(d.debtToEquity / 8).toFixed(4),
    "Current Ratio": +(d.currentRatio / 2).toFixed(4),
    "Quick Ratio": +(d.quickRatio / 2).toFixed(4),
    "P/E": +(d.peRatio).toFixed(2),
    "Price/Sales": +(d.priceToSales).toFixed(2),
    "Stock Price": +(d.stockPrice).toFixed(2),
    CPI: +(250 + Math.random() * 20).toFixed(2),
    "Oil Price": +(70 + Math.random() * 15).toFixed(2),
    "Copper Price": +(3.5 + Math.random() * 1.5).toFixed(2),
    "GDP Growth": +(1.5 + Math.random() * 2).toFixed(2),
    "Fed Funds Rate": +(4.5 + Math.random() * 1).toFixed(2),
    Sentiment: +(0.3 + Math.random() * 0.5).toFixed(4),
    FHI: +d.fhi.toFixed(4),
  }));
}

function emptyRow(cols: string[]): DataRow {
  const row: DataRow = {};
  cols.forEach((c) => { row[c] = ""; });
  return row;
}

export default function PredictPage() {
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [predicted, setPredicted] = useState(false);
  const [dataSource, setDataSource] = useState<"preloaded" | "uploaded" | "manual">("preloaded");
  const [inputRows, setInputRows] = useState<DataRow[]>(defaultRows());
  const [uploadFileName, setUploadFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const isAllFeatures = selectedModel.features.includes("14");
  const featureCols = isAllFeatures ? COLUMNS_14 : COLUMNS_8;
  const allCols = [...featureCols, ...TARGET_COLS];

  const [manualRows, setManualRows] = useState<DataRow[]>(Array.from({ length: 13 }, () => emptyRow(allCols)));

  // Filter pre-loaded data to only include columns relevant to the selected model
  const filteredInputRows = inputRows.map((row) => {
    const filtered: DataRow = {};
    allCols.forEach((c) => { if (c in row) filtered[c] = row[c]; });
    return filtered;
  });

  const activeRows = dataSource === "manual" ? manualRows : filteredInputRows;

  // API response state
  const [predResult, setPredResult] = useState<{
    predicted_fhi: number;
    current_fhi: number;
    change: number;
    change_percent: number;
    health_status: string;
    model_used: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const predFHI = predResult?.predicted_fhi ?? 0;
  const currentFHI = predResult?.current_fhi ?? financialData[financialData.length - 1].fhi;
  const delta = predResult?.change ?? 0;
  const pct = predResult?.change_percent ?? 0;

  const healthLabel = predResult?.health_status ?? (predFHI >= 0.75 ? "Strong" : predFHI >= 0.5 ? "Moderate" : predFHI >= 0.25 ? "Weak" : "Critical");
  const healthColor = predFHI >= 0.75 ? COLORS.green : predFHI >= 0.5 ? COLORS.yellow : predFHI >= 0.25 ? COLORS.orange : COLORS.red;

  const chartData = financialData.slice(-36).map((d) => ({ date: d.date.slice(0, 7), fhi: d.fhi }));
  if (predicted) chartData.push({ date: "2025-05", fhi: predFHI });

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadFileName(file.name);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      const rows = parseCSV(text);
      if (rows.length === 0) {
        toast.error("CSV file is empty or could not be parsed.");
        return;
      }

      // Validate uploaded columns match selected model
      const uploadedCols = Object.keys(rows[0]).filter((c) => c !== "Date" && c !== "FHI");
      const expectedFeatureCols = (isAllFeatures ? COLUMNS_14 : COLUMNS_8).filter((c) => c !== "Date");
      const expectedCount = expectedFeatureCols.length;
      const uploadedCount = uploadedCols.length;

      // Check if columns match the model
      const missingCols = expectedFeatureCols.filter((c) => !uploadedCols.includes(c));
      const extraCols = uploadedCols.filter((c) => !expectedFeatureCols.includes(c));

      if (missingCols.length > 0 || extraCols.length > 0) {
        const modelLabel = isAllFeatures ? "All Features (14)" : "Ratios Only (8)";
        let msg = `Column mismatch for "${selectedModel.name}" (${modelLabel}).`;
        if (missingCols.length > 0) msg += ` Missing: ${missingCols.join(", ")}.`;
        if (extraCols.length > 0) msg += ` Unexpected: ${extraCols.join(", ")}.`;
        msg += ` Expected ${expectedCount} feature columns: ${expectedFeatureCols.join(", ")}.`;
        toast.error(msg, { duration: 8000 });
        // Reset file input
        if (fileRef.current) fileRef.current.value = "";
        setUploadFileName(null);
        return;
      }

      if (rows.length < 13) {
        toast.warning(`Uploaded ${rows.length} rows — minimum 13 required (12 lookback + 1 for differencing). Prediction may fail.`, { duration: 6000 });
      }

      setInputRows(rows);
      setDataSource("uploaded");
      setPredicted(false);
      toast.success(`Loaded ${rows.length} rows with ${uploadedCount} features — matches ${selectedModel.name}.`);
    };
    reader.readAsText(file);
  }, [isAllFeatures, selectedModel]);

  const updateManualCell = (rowIdx: number, col: string, value: string) => {
    setManualRows((prev) => {
      const next = [...prev];
      next[rowIdx] = { ...next[rowIdx], [col]: value };
      return next;
    });
  };

  const addManualRow = () => setManualRows((prev) => [...prev, emptyRow(allCols)]);
  const removeManualRow = (idx: number) => setManualRows((prev) => prev.filter((_, i) => i !== idx));

  const handlePredict = async () => {
    // Convert UI rows to API format
    const cols = isAllFeatures ? COLUMNS_14 : COLUMNS_8;
    const featureKeys = cols.filter((c) => c !== "Date");

    // Validate: check rows have actual data filled in (not empty strings)
    const filledRows = activeRows.filter((row) => {
      return featureKeys.every((col) => {
        const val = String(row[col] ?? "").trim();
        return val !== "" && !isNaN(parseFloat(val));
      });
    });

    if (filledRows.length < 13) {
      toast.error(`Need at least 13 fully filled rows (got ${filledRows.length}). The model requires 12 lookback + 1 for differencing. Please fill all ${featureKeys.length} feature columns in each row.`);
      return;
    }

    // For manual/uploaded: validate that rows only have the correct features (8 or 14 rule)
    if (dataSource === "manual" || dataSource === "uploaded") {
      const rowCols = Object.keys(activeRows[0]).filter((c) => c !== "Date" && c !== "FHI");
      const expectedCols = featureKeys;
      const missingCols = expectedCols.filter((c) => !rowCols.includes(c));
      const extraCols = rowCols.filter((c) => !expectedCols.includes(c) && c !== "Stock Price");
      
      if (missingCols.length > 0) {
        toast.error(`Data mismatch for "${selectedModel.name}". Missing columns: ${missingCols.join(", ")}. This model requires ${featureKeys.length} features.`);
        return;
      }
      if (extraCols.length > 0) {
        toast.error(`Data mismatch for "${selectedModel.name}". Unexpected columns: ${extraCols.join(", ")}. This model expects only ${featureKeys.length} features.`);
        return;
      }
    }
    
    const apiFeatures = filledRows.map((row) => {
      const apiRow: Record<string, number> = {};
      featureKeys.forEach((col) => {
        const apiName = COL_TO_API[col] || col;
        const val = parseFloat(String(row[col]));
        apiRow[apiName] = isNaN(val) ? 0 : val;
      });
      return apiRow;
    });

    setLoading(true);
    setPredicted(false);
    setPredResult(null);

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: selectedModel.apiType,
          features: apiFeatures,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
      }

      const data = await resp.json();
      setPredResult(data);
      setPredicted(true);
      toast.success(`Prediction complete using ${data.model_used}`);
    } catch (e: any) {
      toast.error(`Prediction failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderDataTable = (rows: DataRow[], cols: string[]) => (
    <div className="overflow-x-auto scrollbar-thin" style={{ maxHeight: 400 }}>
      <table className="w-full text-xs font-mono min-w-[900px]">
        <thead className="sticky top-0 z-10" style={{ background: "#161b22" }}>
          <tr className="border-b border-border">
            {cols.map((c) => (
              <th key={c} className="p-2 text-left text-muted-foreground whitespace-nowrap">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border/30 hover:bg-secondary/30">
              {cols.map((c) => (
                <td key={c} className="p-2 text-foreground whitespace-nowrap">
                  {typeof row[c] === "number" ? (row[c] as number).toFixed?.(4) ?? row[c] : row[c] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderManualTable = () => {
    const cols = allCols;
    return (
      <div className="overflow-x-auto scrollbar-thin" style={{ maxHeight: 420 }}>
        <table className="w-full text-xs font-mono min-w-[1100px]">
          <thead className="sticky top-0 z-10" style={{ background: "#161b22" }}>
            <tr className="border-b border-border">
              <th className="p-1 text-muted-foreground w-8">#</th>
              {cols.map((c) => (
                <th key={c} className="p-1 text-left text-muted-foreground whitespace-nowrap">{c}</th>
              ))}
              <th className="p-1 w-8" />
            </tr>
          </thead>
          <tbody>
            {manualRows.map((row, i) => (
              <tr key={i} className="border-b border-border/30">
                <td className="p-1 text-muted-foreground text-center">{i + 1}</td>
                {cols.map((c) => (
                  <td key={c} className="p-1">
                    <Input
                      value={row[c] ?? ""}
                      onChange={(e) => updateManualCell(i, c, e.target.value)}
                      className="h-7 text-xs font-mono bg-secondary/50 border-border/50 px-1.5 min-w-[80px]"
                      placeholder={c}
                    />
                  </td>
                ))}
                <td className="p-1">
                  <button
                    onClick={() => removeManualRow(i)}
                    className="text-muted-foreground hover:text-destructive transition-colors text-sm px-1"
                    title="Remove row"
                  >
                    x
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div>
      <h1 className="text-2xl font-semibold text-foreground mb-1">Predict Next-Month FHI</h1>
      <p className="text-muted-foreground text-sm mb-6">
        Runs the <strong className="text-foreground">exact same inference pipeline</strong> as the training notebook — same scaler, same feature columns, same sequence construction.
      </p>

      <SectionHeader title="1 — Select Model" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => {
              setSelectedModel(m);
              setPredicted(false);
              setPredResult(null);
              const cols = m.features.includes("14") ? [...COLUMNS_14, ...TARGET_COLS] : [...COLUMNS_8, ...TARGET_COLS];
              setManualRows(Array.from({ length: 13 }, () => emptyRow(cols)));
            }}
            className={`card-dashboard text-left transition-all ${
              selectedModel.id === m.id ? "border-primary/50 ring-1 ring-primary/30" : "hover:border-muted-foreground/30"
            }`}
          >
            <div className="card-title-label">{m.type.toUpperCase()}</div>
            <div className="text-sm text-foreground">{m.features}</div>
            <div className="text-xs text-muted-foreground mt-1 font-mono">
              {m.type === "lstm" ? "50 units | LB=12" : m.features.includes("All") ? "(3,1,4)" : "(2,1,4)"}
            </div>
          </button>
        ))}
      </div>

      <div className="card-dashboard mb-6 font-mono text-xs text-muted-foreground leading-7">
        <span className="text-foreground font-medium">Architecture:</span> {selectedModel.type === "lstm" ? "LSTM (deep learning)" : "ARIMAX (statistical)"}<br />
        <span className="text-foreground font-medium">Feature columns:</span> {selectedModel.features}<br />
        <span className="text-foreground font-medium">LSTM input shape:</span> (batch=1, lookback=12, features={selectedModel.features.includes("14") ? 14 : 8})
      </div>

      <SectionHeader title="2 — Input Data" />

      <Tabs
        value={dataSource}
        onValueChange={(v) => {
          setDataSource(v as "preloaded" | "uploaded" | "manual");
          setPredicted(false);
          if (v === "preloaded") setInputRows(defaultRows());
        }}
        className="mb-6"
      >
        <TabsList className="bg-secondary/50 border border-border mb-4">
          <TabsTrigger value="preloaded" className="text-xs data-[state=active]:bg-primary/20">Pre-loaded Data</TabsTrigger>
          <TabsTrigger value="uploaded" className="text-xs data-[state=active]:bg-primary/20">Upload CSV</TabsTrigger>
          <TabsTrigger value="manual" className="text-xs data-[state=active]:bg-primary/20">Manual Entry</TabsTrigger>
        </TabsList>

        <TabsContent value="preloaded">
          <div className="card-dashboard">
            <div className="text-xs text-muted-foreground mb-3">
              Uses the scaled dataset already loaded — the <strong className="text-foreground">same scaler</strong> fitted on the training split. Scroll horizontally to see all columns.
            </div>
            {renderDataTable(inputRows, allCols)}
            <p className="text-xs text-muted-foreground mt-2">These {inputRows.length} rows form the prediction window.</p>
          </div>
        </TabsContent>

        <TabsContent value="uploaded">
          <div className="card-dashboard">
            <div className="text-xs text-muted-foreground mb-3">
              Upload a CSV with <strong className="text-foreground">{isAllFeatures ? "14" : "8"} feature columns</strong> matching the <strong className="text-foreground">{selectedModel.name}</strong> model.
              <br />Expected columns: <span className="font-mono text-[10px]">{featureCols.filter(c => c !== "Date").join(", ")}</span>
            </div>
            <div className="flex items-center gap-3 mb-4">
              <input
                ref={fileRef}
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => fileRef.current?.click()}
                className="text-xs"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Choose CSV File
              </Button>
              {uploadFileName && (
                <span className="text-xs text-muted-foreground font-mono">{uploadFileName}</span>
              )}
            </div>
            {dataSource === "uploaded" && inputRows.length > 0 && (
              <>
                {renderDataTable(inputRows, Object.keys(inputRows[0]))}
                <p className="text-xs text-muted-foreground mt-2">{inputRows.length} rows loaded from file.</p>
              </>
            )}
            {dataSource === "uploaded" && inputRows.length === 0 && (
              <p className="text-xs text-muted-foreground">No data loaded yet. Upload a CSV file.</p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="manual">
          <div className="card-dashboard">
            <div className="text-xs text-muted-foreground mb-3">
              Enter your own data below. Model <strong className="text-foreground">{selectedModel.name}</strong> requires <strong className="text-foreground">{isAllFeatures ? "14" : "8"} features</strong> with at least <strong className="text-foreground">13 fully filled rows</strong>.
              <br />Required columns: <span className="font-mono text-[10px]">{featureCols.filter(c => c !== "Date").join(", ")}</span>
            </div>
            {renderManualTable()}
            <div className="flex items-center gap-3 mt-3">
              <Button variant="outline" size="sm" onClick={addManualRow} className="text-xs">
                + Add Row
              </Button>
              <span className="text-xs text-muted-foreground">{manualRows.length} rows</span>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      <SectionHeader title="3 — Run Prediction" />
      <div className="text-xs text-muted-foreground mb-3">
        Source: <strong className="text-foreground">{dataSource === "preloaded" ? "Pre-loaded historical data" : dataSource === "uploaded" ? `Uploaded: ${uploadFileName}` : "Manual entry"}</strong> · {activeRows.length} rows · Model: <strong className="text-foreground">{selectedModel.apiType}</strong> · Min 13 rows required
      </div>
      <Button onClick={handlePredict} className="mb-6" disabled={loading}>
        {loading ? (
          <>
            <svg className="animate-spin mr-1.5" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" strokeOpacity="0.25"/><path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round"/></svg>
            Predicting…
          </>
        ) : (
          <>
            <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor" className="mr-1"><path d="M0 0 L12 7 L0 14Z"/></svg>
            Run Prediction
          </>
        )}
      </Button>

      {predicted && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6 mb-6">
            <div className="card-dashboard text-center" style={{ borderTop: `3px solid ${healthColor}` }}>
              <div className="card-title-label">Predicted FHI</div>
              <div className="font-mono text-4xl font-bold leading-tight" style={{ color: healthColor }}>{predFHI.toFixed(4)}</div>
              <div className="text-sm font-semibold mt-1" style={{ color: healthColor }}>{healthLabel}</div>
            </div>
            <div className="card-dashboard text-center" style={{ borderTop: `3px solid ${COLORS.primary}` }}>
              <div className="card-title-label">Current FHI</div>
              <div className="font-mono text-4xl font-bold text-foreground leading-tight">{currentFHI.toFixed(4)}</div>
              <div className="text-sm text-muted-foreground mt-1">Latest data point</div>
            </div>
            <div className="card-dashboard text-center" style={{ borderTop: `3px solid ${delta >= 0 ? COLORS.green : COLORS.red}` }}>
              <div className="card-title-label">Month-on-Month</div>
              <div className="font-mono text-4xl font-bold leading-tight" style={{ color: delta >= 0 ? COLORS.green : COLORS.red }}>
                {delta >= 0 ? "+" : "-"} {Math.abs(delta).toFixed(4)}
              </div>
              <div className="text-sm mt-1" style={{ color: delta >= 0 ? COLORS.green : COLORS.red }}>{pct >= 0 ? "+" : ""}{pct}%</div>
            </div>
            <div className="card-dashboard text-center" style={{ borderTop: `3px solid ${COLORS.muted}` }}>
              <div className="card-title-label">Model / Input</div>
              <div className="font-mono text-lg font-semibold text-foreground mt-2">{selectedModel.type.toUpperCase()}</div>
              <div className="text-xs text-muted-foreground mt-1">{selectedModel.features}</div>
            </div>
          </div>

          <div className="card-dashboard mb-6">
            <div className="text-xs text-muted-foreground mb-4 uppercase tracking-wider">FHI Gauge</div>
            <div className="relative h-8 bg-secondary rounded-full overflow-hidden">
              <div className="absolute inset-y-0 left-0 w-1/4 bg-destructive/15" />
              <div className="absolute inset-y-0 left-1/4 w-1/4" style={{ background: "rgba(255,123,114,0.1)" }} />
              <div className="absolute inset-y-0 left-2/4 w-1/4" style={{ background: "rgba(210,153,34,0.1)" }} />
              <div className="absolute inset-y-0 left-3/4 w-1/4 bg-success/10" />
              <div className="absolute top-0 h-full w-1 transition-all duration-700" style={{ left: `${predFHI * 100}%`, background: healthColor }} />
              <div className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full transition-all duration-700" style={{ left: `calc(${predFHI * 100}% - 6px)`, background: healthColor, boxShadow: `0 0 8px ${healthColor}` }} />
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1 font-mono">
              <span>0.00 Critical</span><span>0.25 Weak</span><span>0.50 Moderate</span><span>0.75 Strong</span><span>1.00</span>
            </div>
          </div>

          <div className="card-dashboard">
            <div className="text-xs text-muted-foreground mb-3 uppercase tracking-wider">FHI History + Next-Month Forecast</div>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.1} />
                    <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                <XAxis dataKey="date" stroke={COLORS.muted} tick={{ fontSize: 10 }} interval={4} />
                <YAxis stroke={COLORS.muted} tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="fhi" stroke={COLORS.primary} strokeWidth={2.5} fill="url(#predGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <InsightBox>
            <strong>{predResult?.model_used ?? selectedModel.name}</strong> predicts an FHI of <strong style={{ color: healthColor }}>{predFHI.toFixed(4)}</strong> for next month — a {delta >= 0 ? "increase" : "decrease"} of {Math.abs(delta).toFixed(4)} ({Math.abs(pct).toFixed(2)}%) from the current {currentFHI.toFixed(4)}. Health: <strong style={{ color: healthColor }}>{healthLabel}</strong>.
          </InsightBox>
        </div>
      )}
    </div>
  );
}
