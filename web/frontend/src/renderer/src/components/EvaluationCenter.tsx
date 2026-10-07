/**
 * EvaluationCenter.tsx - system-level evaluation workspace.
 *
 * Evaluation measures development changes and runtime quality, so it lives
 * beside the configuration centre rather than beside business workspaces.
 */

import { useEffect, useState, type FC } from 'react';
import { EvalPanel } from './EvalPanel';

export type EvalSection = 'single' | 'runs';

interface Props {
  onClose: () => void;
}

const NAV: Array<{
  id: EvalSection;
  icon: string;
  label: string;
  description: string;
}> = [
  {
    id: 'single',
    icon: '◎',
    label: '单次诊断',
    description: '时间线、证据与失败链',
  },
  {
    id: 'runs',
    icon: '▦',
    label: '数据集评测',
    description: '门禁、回归与 badcase',
  },
];

export const EvaluationCenter: FC<Props> = ({ onClose }) => {
  const [section, setSection] = useState<EvalSection>('single');

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="eval-center-backdrop"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 120,
        padding: '3vh 2.5vw',
        background: 'rgba(15, 23, 42, 0.52)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        className="eval-center-shell"
        onClick={event => event.stopPropagation()}
        style={{
          width: 'min(1440px, 98vw)',
          height: 'min(94vh, 960px)',
          minHeight: 660,
          background: 'var(--color-bg)',
          border: '1px solid var(--color-border)',
          borderRadius: 12,
          boxShadow: 'var(--shadow-md)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <header
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '12px 16px',
            borderBottom: '1px solid var(--color-border)',
            background: 'var(--color-surface)',
            flexShrink: 0,
          }}
        >
          <span className="eval-center-mark">EV</span>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>评测中心</div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 1 }}>
              从发布结论下钻到样本与 trace
            </div>
          </div>
          <div style={{ flex: 1 }} />
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 999,
              fontSize: 11,
              color: 'var(--color-text-secondary)',
              background: 'var(--color-inset)',
              border: '1px solid var(--color-border)',
            }}
          >
            决策工作台
          </span>
          <button
            onClick={onClose}
            title="关闭 (Esc)"
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              color: 'var(--color-text-secondary)',
              fontSize: 14,
            }}
          >
            ×
          </button>
        </header>

        <div style={{ flex: 1, minHeight: 0, display: 'flex', overflow: 'hidden' }}>
          <nav
            style={{
              width: 176,
              flexShrink: 0,
              padding: '12px 9px',
              borderRight: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
            }}
          >
            <div
              style={{
                padding: '0 8px 8px',
                fontSize: 10,
                fontWeight: 700,
                color: 'var(--color-text-tertiary)',
              }}
            >
              EVALUATION
            </div>
            {NAV.map(item => {
              const active = section === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setSection(item.id)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 9,
                    padding: '10px 9px',
                    marginBottom: 6,
                    borderRadius: 8,
                    textAlign: 'left',
                    border: active
                      ? '1px solid var(--color-primary)'
                      : '1px solid transparent',
                    background: active ? 'var(--color-primary-light)' : 'transparent',
                    color: active ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                  }}
                >
                  <span
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: 6,
                      display: 'grid',
                      placeItems: 'center',
                      flexShrink: 0,
                      background: active ? 'var(--color-primary)' : 'var(--color-inset)',
                      color: active ? '#fff' : 'var(--color-text-secondary)',
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    {item.icon}
                  </span>
                  <span style={{ minWidth: 0 }}>
                    <span style={{ display: 'block', fontSize: 13, fontWeight: 600 }}>
                      {item.label}
                    </span>
                    <span
                      style={{
                        display: 'block',
                        fontSize: 10,
                        lineHeight: 1.4,
                        marginTop: 2,
                        color: 'var(--color-text-tertiary)',
                      }}
                    >
                      {item.description}
                    </span>
                  </span>
                </button>
              );
            })}
            <div
              style={{
                marginTop: 14,
                padding: '10px 9px',
                borderRadius: 8,
                border: '1px dashed var(--color-border)',
                color: 'var(--color-text-tertiary)',
                fontSize: 10,
                lineHeight: 1.55,
              }}
            >
              先看发布结论，再按案例与 trace 归因；指标、配置和门禁保持同屏审阅。
            </div>
          </nav>

          <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
            <EvalPanel section={section} onSectionChange={setSection} hideSectionNav />
          </main>
        </div>
      </div>
    </div>
  );
};

export default EvaluationCenter;
