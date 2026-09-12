import type { ClusterMode, ColorBy, SizeBy } from './graphScene';

export type LabelMode = 'off' | 'auto' | 'all';

/** Everything the in-canvas controls own. One object so a control change is one state update. */
export interface GraphViewState {
  /** Whether communities are collapsed. Independent of how nodes are coloured. */
  clusterMode: ClusterMode;
  resolution: number;
  colorBy: ColorBy;
  sizeBy: SizeBy;
  showHulls: boolean;
  labelMode: LabelMode;
  showEdges: boolean;
  curvedEdges: boolean;
  /** 0 means focus is off; otherwise how many hops around the selected node stay visible. */
  focusHops: number;
  pathMode: boolean;
  search: string;
}

export const DEFAULT_VIEW_STATE: GraphViewState = {
  clusterMode: 'off',
  resolution: 1,
  colorBy: 'label',
  sizeBy: 'degree',
  showHulls: false,
  labelMode: 'auto',
  showEdges: true,
  curvedEdges: false,
  focusHops: 0,
  pathMode: false,
  search: '',
};
