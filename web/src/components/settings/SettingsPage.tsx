import { useCallback, useEffect, useMemo, useState } from 'react';
import { Apparatus, Rail, Screen } from '../../design/parts';
import { changeLocalAiIntegration, fetchLocalAiIntegrations } from '../../lib/api';
import { errorMessage } from '../../lib/format';
import type { LocalAiIntegration } from '../../types';

type IntegrationAction = 'install' | 'update' | 'repair';

function actionFor(integration: LocalAiIntegration): IntegrationAction | undefined {
  if (!integration.detected) return undefined;
  if (integration.state === 'not-installed') return 'install';
  if (integration.state === 'update-available') return 'update';
  if (integration.state === 'repair-required') return 'repair';
  return undefined;
}

function stateLabel(integration: LocalAiIntegration): string {
  if (!integration.detected) return 'Not detected';
  switch (integration.state) {
    case 'current': return 'Installed';
    case 'activation-required': return 'Installed · activation required';
    case 'update-available': return 'Update available';
    case 'repair-required': return 'Repair required';
    case 'newer-than-runtime': return 'Newer than IronGraph';
    default: return 'Ready to install';
  }
}

export function SettingsPage() {
  const [integrations, setIntegrations] = useState<LocalAiIntegration[]>([]);
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState<string>();
  const [error, setError] = useState<string>();
  const [message, setMessage] = useState<string>();

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      setIntegrations(await fetchLocalAiIntegrations(signal));
      setError(undefined);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setError(errorMessage(cause));
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const ordered = useMemo(
    () => [...integrations].sort((left, right) => Number(right.detected) - Number(left.detected) || left.display_name.localeCompare(right.display_name)),
    [integrations],
  );
  const detected = integrations.filter((integration) => integration.detected).length;
  const installed = integrations.filter((integration) => integration.state !== 'not-installed' && integration.state !== 'repair-required').length;

  const change = async (integration: LocalAiIntegration, action: IntegrationAction) => {
    setActing(integration.host);
    setError(undefined);
    setMessage(undefined);
    try {
      setIntegrations(await changeLocalAiIntegration(integration.host, action));
      const result = action === 'repair' ? 'installation repaired' : action === 'update' ? 'package updated' : 'integration installed';
      const activation = integration.activation_instruction ?? 'Restart the host and begin a new conversation.';
      setMessage(integration.display_name + ': ' + result + '. ' + activation);
    } catch (cause) {
      setError(errorMessage(cause));
      await refresh();
    } finally {
      setActing(undefined);
    }
  };

  return (
    <Screen name="settings">
      <nav className="col-i settings-index" aria-label="Settings sections">
        <div className="settings-index-head"><span className="ap faint">Settings</span><h2>Local connections</h2></div>
        <button className="settings-index-item on" type="button"><span className="ap">01</span><span><b>AI hosts</b><small>Skills, plugins, and MCP</small></span></button>
        <div className="settings-index-facts">
          <span><b>{detected}</b><small>hosts detected</small></span>
          <span><b>{installed}</b><small>integrations installed</small></span>
        </div>
      </nav>
      <Rail screen="settings" cap="Settings" keys={[{ t: '01', on: true, ticks: integrations.map((integration) => integration.detected) }]} foot="Local" />
      <article className="col-ii settings-work">
        <svg className="rd-over" aria-hidden></svg>
        <header className="settings-heading">
          <p className="ap origin">Settings · AI hosts</p>
          <h1>Use IronGraph from your assistants</h1>
          <p>Choose where IronGraph should act as durable graph memory. Installation adds the strongest package each detected host supports: plugin where possible, otherwise MCP with an Agent Skill or host-native guidance.</p>
          <div className="bar bare"><span className="ap faint">Package version follows this IronGraph build</span><span className="grow"></span><button className="detent" type="button" onClick={() => void refresh()} disabled={loading}>{loading ? 'Detecting…' : 'Detect again'}</button></div>
        </header>

        {message && <div className="notice" role="status"><span className="kindmark"></span><span className="ap lbl">Complete</span><p>{message}</p></div>}
        {error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Installation failed</span><p>{error}</p></div>}
        {loading && integrations.length === 0 ? <p className="empty-line">Looking for local AI hosts…</p> : (
          <div className="settings-integrations" aria-label="Local AI integrations">
            {ordered.map((integration) => {
              const action = actionFor(integration);
              const working = acting === integration.host;
              const workingLabel = action === 'repair' ? 'Repairing…' : action === 'update' ? 'Updating…' : 'Installing…';
              const actionLabel = action ? action[0]!.toUpperCase() + action.slice(1) : integration.detected ? 'Installed' : 'Not found';
              return <article className={'settings-integration' + (integration.detected ? ' detected' : '')} key={integration.host}>
                <span className={integration.detected ? 'pip' : 'pip hollow'}></span>
                <div className="settings-integration-name"><strong>{integration.display_name}</strong><small>{integration.integration}</small></div>
                <div className="settings-integration-version"><span className="ap faint">{stateLabel(integration)}</span><small>{integration.installed_version ? 'v' + integration.installed_version : integration.detected ? 'v' + integration.available_version + ' available' : 'Application unavailable'}</small></div>
                <button className="detent" type="button" disabled={!action || Boolean(acting)} onClick={() => action && void change(integration, action)}>{working ? workingLabel : actionLabel}</button>
              </article>;
            })}
          </div>
        )}

      </article>
      <Apparatus heading="Integration" notes={[
        { lbl: 'Detected only', origin: true, p: 'IronGraph never injects a new integration merely because a host application exists. You choose Install.' },
        { lbl: 'Versioned', p: 'Previously managed packages refresh when IronGraph starts. A missing managed file changes the action to Repair.' },
        { lbl: 'Local boundary', p: 'These controls exist only on the plain loopback console. Remote Query listeners do not expose them.' },
      ]} />
    </Screen>
  );
}
