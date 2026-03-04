/**
 * Minimal recharts mock for Jest.
 * All chart components render a simple div placeholder.
 */
const React = require("react");

const createMock = (name) => {
  const Comp = ({ children, ...props }) =>
    React.createElement("div", { "data-testid": `recharts-${name}` }, children);
  Comp.displayName = name;
  return Comp;
};

module.exports = {
  ResponsiveContainer: createMock("ResponsiveContainer"),
  RadialBarChart: createMock("RadialBarChart"),
  RadialBar: createMock("RadialBar"),
  PolarAngleAxis: createMock("PolarAngleAxis"),
  BarChart: createMock("BarChart"),
  Bar: createMock("Bar"),
  Cell: createMock("Cell"),
  XAxis: createMock("XAxis"),
  YAxis: createMock("YAxis"),
  CartesianGrid: createMock("CartesianGrid"),
  Tooltip: createMock("Tooltip"),
  ReferenceLine: createMock("ReferenceLine"),
  ScatterChart: createMock("ScatterChart"),
  Scatter: createMock("Scatter"),
  ReferenceDot: createMock("ReferenceDot"),
  AreaChart: createMock("AreaChart"),
  Area: createMock("Area"),
  LineChart: createMock("LineChart"),
  Line: createMock("Line"),
};
