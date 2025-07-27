# devsecops-security-anomaly
The technology to identify security anomalies in enterprise DevSecOps platforms. This is provided under the BSD-3 license and supporting commercial & academic research.

## Structure

This repository will provide iterations of research and snapshots of artifacts at the time of publication.

- library: foundational modules for use across the repo
- ieee-secdev-2025: implementation supporting daily analysis of anomalies in Enterprise GitHub that combines global and local features, including user and entity behavior analytics.

## IEEE SecDev 2025: User Entity Behavior Analytics (UEBA) Enhanced Security Anomaly Detection in Enterprise DevSecOps Platforms

The source under *ueba-security-anomalies* is the artifacts supplementing these conference proceedings to exemplify detection of security anomalies within a specific implementation of Enterprise GitHub.

For details on building & running, refer to the local [ieee-secdev-2025/README](ieee-secdev-2025/README.md).

**Abstract**
*Secure software delivery platforms are essential for the management, release, and deployment of software. They generate billions of audit events, yet most anomaly detectors view all users through a single organizational lens. We introduce a dual‑granularity framework that fuses global outlier mining with user and entity behavior analytics (UEBA) to surface both blatant abuses and subtle deviations inside Enterprise GitHub. Leveraging 403 million audit log records spanning 147 days and 15K user, we engineer security‑centered features in three layers: raw action counts, temporal/IP enrichments, and per‑user analysis that capture history‑aware drift. An Isolation Forest tuned via contamination back-analysis isolates candidate threats in minutes without labeled data. Daily scoring in a Fortune 100 system flagged 26 anomalies: filtering 99.99\% of activity reveals exfiltration‑style repository downloads, privilege bypasses, and policy overrides hours after they occurred. In contrast, a global‑only baseline produced 64 alerts dominated by noisy service accounts. The proposed blend of UEBA and global metrics therefore cuts analyst workload while exposing insider‑level anomalies previously hidden beneath organizational norms. Because the method relies solely on native platform logs and open source tools, it can be transplanted to any large‑scale DevOps environment to harden the last mile of the software supply chain.*