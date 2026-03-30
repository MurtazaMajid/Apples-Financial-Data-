import { TrendingUp, TrendingDown } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
  accentColor?: string;
}

export default function MetricCard({ label, value, delta, deltaPositive }: MetricCardProps) {
  return (
    <div className="card-dashboard animate-fade-in flex flex-col items-center justify-between text-center h-full min-h-[130px] py-5">
      <div className="card-title-label">{label}</div>
      <div className="card-value">{value}</div>
      <div className="h-5 flex items-center justify-center">
        {delta ? (
          <div className={`flex items-center justify-center gap-1 text-xs font-mono ${deltaPositive ? "text-success" : "text-destructive"}`}>
            {deltaPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {delta}
          </div>
        ) : null}
      </div>
    </div>
  );
}
