## Overview
This binary detects faces in images from disk or live camera feed, extracts distinct facial signatures, and publishes unique detection events to a message bus. The goal of this binary is to perform person identification in a way which allows for the application of labels after detection and perform the appropriate response action, which will vary per application. My first use case is to tackle a personal problem of mine (not signing in at the gym) by automatically detecting myself on gym security camera footage and sending an API request. Eventually, this may be extended to other users easily via an opt-in approach, where a user may _e.g._, supply an image of themselves which may then be used to generate their reference embedding and automatically determine classes they attend--even retroactively.

``` mermaid
---
title: Application Workflow
---
flowchart
    ce[Client Enters] --> fd[Detect Face]
    fd --> ge[Generate Embedding]
    ge --> kc{Known Client?}
    kc --> |Yes| pe[Publish Event]
    kc --> |No| p{Privacy?}
    p --> |Yes| d[Discard]
    p --> |No| tr[Track Without Label]
    pe --> ar[Asynchronous Action]
```

## Roadmap
- [ ] Limit data generation (_e.g._, do not publish similar embeddings in a given time frame)
- [ ] Add publishing to RabbitMQ message bus
- [ ] Move data visualization to a separate binary
- [ ] Generate Docker Compose and/or Helm charts for Postgresql w/ Vector, application, and RabbitMQ
- [ ] Improve startup and shutdown robustness (Nokhwa crate often has issues with initial webcam setup and rude shutdown)
