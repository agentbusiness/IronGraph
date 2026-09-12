import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import { applyStoredTheme } from './hooks/useTheme';
// The design's two voices, bundled so the console reads the same on every machine. The stylesheet
// names them with system fallbacks, so a build without these files still opens — in a worse hand.
import '@fontsource/archivo/400.css';
import '@fontsource/archivo/500.css';
import '@fontsource/archivo/600.css';
import '@fontsource/archivo/700.css';
import '@fontsource/source-serif-4/400.css';
import '@fontsource/source-serif-4/400-italic.css';
import '@fontsource/source-serif-4/600.css';
import './reference.css';
import './design/components.css';
import './console.css';

// Before the first render, so the first paint and any failure that happens instead of a render
// are both on the reader's ground rather than the stylesheet's default.
applyStoredTheme();

const root = document.getElementById('root');
if (!root) throw new Error('Application root is missing.');

createRoot(root).render(
  <StrictMode>
    <ErrorBoundary><App /></ErrorBoundary>
  </StrictMode>,
);
