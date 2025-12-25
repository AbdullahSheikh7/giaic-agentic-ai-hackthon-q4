import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="text--center">
          <Heading as="h1" className="hero__title">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started - Agentic AI Guide
          </Link>
          <Link
            className="button button--primary button--lg margin-left--md"
            to="/docs/protocols/intro">
            Explore Protocols
          </Link>
        </div>
      </div>
    </header>
  );
}

function HomepageCards() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--4">
            <div className="card-demo">
              <div className="card">
                <div className="card__header">
                  <h3>Agentic AI</h3>
                </div>
                <div className="card__body">
                  <p>Learn about agents, architectures, and implementation patterns.</p>
                </div>
                <div className="card__footer">
                  <Link className="button button--primary button--block" to="/docs/intro">
                    Start Learning
                  </Link>
                </div>
              </div>
            </div>
          </div>
          <div className="col col--4">
            <div className="card-demo">
              <div className="card">
                <div className="card__header">
                  <h3>Protocol Standards</h3>
                </div>
                <div className="card__body">
                  <p>Deep dive into MCP, A2A, HTTP, and other communication protocols.</p>
                </div>
                <div className="card__footer">
                  <Link className="button button--primary button--block" to="/docs/protocols/intro">
                    Explore Protocols
                  </Link>
                </div>
              </div>
            </div>
          </div>
          <div className="col col--4">
            <div className="card-demo">
              <div className="card">
                <div className="card__header">
                  <h3>AI Projects</h3>
                </div>
                <div className="card__body">
                  <p>Real-world projects and implementations using AI agents.</p>
                </div>
                <div className="card__footer">
                  <Link className="button button--primary button--block" to="/docs/projects/intro">
                    View Projects
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Comprehensive guide to AI agents, protocols, and implementation">
      <HomepageHeader />
      <main>
        <HomepageCards />
      </main>
    </Layout>
  );
}
