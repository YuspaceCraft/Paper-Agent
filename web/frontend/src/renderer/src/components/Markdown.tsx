/**
 * Markdown.tsx — markdown renderer for assistant messages.
 *
 * ponytail: reuse the already-installed react-markdown + react-syntax-highlighter
 * (no markdown-it / tailwind). No math plugin is installed, so katex is skipped.
 */

import type { FC, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import oneLight from 'react-syntax-highlighter/dist/esm/styles/prism/one-light';

interface Props {
  content: string;
}

export const Markdown: FC<Props> = ({ content }) => {
  return (
    <div className="msg-content">
      <ReactMarkdown
        components={{
          code(props) {
            const { className, children } = props as { className?: string; children?: ReactNode };
            const match = /language-(\w+)/.exec(className || '');
            if (match) {
              return (
                <div className="code-block-wrapper">
                  <SyntaxHighlighter
                    language={match[1]}
                    style={oneLight}
                    customStyle={{ margin: 0, borderRadius: 0, fontSize: '12.5px', lineHeight: 1.6, background: 'transparent' }}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              );
            }
            return <code className={className}>{children}</code>;
          },
          a(props) {
            const { href, children } = props as { href?: string; children?: ReactNode };
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;
