import { ShieldAlert, AlertTriangle, Info } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

interface RiskWarningsProps {
  warnings: string[];
  className?: string;
}

/**
 * Displays risk warning banners below a recommendation.
 * High-severity warnings (containing "CRITICAL", "HIGH") get destructive styling.
 */
export function RiskWarnings({ warnings, className }: RiskWarningsProps) {
  if (!warnings.length) return null;

  return (
    <div className={cn("space-y-2", className)}>
      {warnings.map((warning, i) => {
        const isHigh =
          warning.toUpperCase().includes("CRITICAL") ||
          warning.toUpperCase().includes("HIGH");
        const isMedium = warning.toUpperCase().includes("MEDIUM");

        const variant = isHigh ? "destructive" : isMedium ? "warning" : "default";
        const Icon = isHigh ? ShieldAlert : isMedium ? AlertTriangle : Info;

        return (
          <Alert key={i} variant={variant as "destructive" | "warning" | "default"}>
            <Icon className="h-4 w-4" />
            <AlertTitle className="text-xs font-semibold uppercase tracking-wider">
              Risk Warning
            </AlertTitle>
            <AlertDescription className="text-sm">{warning}</AlertDescription>
          </Alert>
        );
      })}
    </div>
  );
}
