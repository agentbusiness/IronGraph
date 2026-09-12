import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Cross, Frame } from '../design/parts';
import { storedTheme } from '../hooks/useTheme';

interface Props { children: ReactNode }
interface State { error?: Error }

export class ErrorBoundary extends Component<Props, State> {
  override state: State = {};

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Uncaught interface error', error, info.componentStack);
  }

  override render(): ReactNode {
    if (this.state.error) {
      return (
        // The reader's ground, read from storage rather than left to the frame's default. This
        // draws when the app has stopped — which is exactly when the theme the app applied can no
        // longer be relied on — and a stop that arrives on the opposite ground reads as a second,
        // larger failure than the one that happened.
        <Frame theme={storedTheme()}>
          <div className="gate" role="alert">
            <span className="ap warn">Stopped</span>
            <h1 className="pagekey">The interface stopped</h1>
            <div className="hr"><Cross /></div>
            {/*
              * The record is not what broke. This is the browser giving up on drawing a page —
              * nothing was being written, and nothing is lost by reloading. Saying so is the
              * difference between an inconvenience and a fright.
              */}
            <p className="prose">
              Something in the page failed while it was being drawn. Your record is untouched: this
              is the interface, not the database, and reloading costs nothing.
            </p>
            <p className="ap faint gate-note">{this.state.error.message}</p>
            <div className="sign">
              <button className="detent" type="button" aria-pressed onClick={() => window.location.reload()}>
                Reload the interface
              </button>
            </div>
          </div>
        </Frame>
      );
    }
    return this.props.children;
  }
}
