export function perspectiveFitDistance(
  width: number,
  height: number,
  depth: number,
  aspect: number,
  verticalFovDegrees: number,
  padding = 1.2,
): number {
  const verticalTangent = Math.tan(Math.max(1, Math.min(179, verticalFovDegrees)) * Math.PI / 360);
  const horizontalTangent = verticalTangent * Math.max(0.01, aspect);
  const planarDistance = Math.max(
    Math.max(0, height) / 2 / verticalTangent,
    Math.max(0, width) / 2 / horizontalTangent,
  );
  return (planarDistance + Math.max(0, depth) / 2) * Math.max(1, padding);
}
