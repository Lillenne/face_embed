## Overview
### Summary
This application detects faces in images from a live camera feed, extracts distinct facial signatures, and publishes unique detection events to a message bus. 
### Goal
Perform person identification while allowing for labels to be created _after_ detection and perform the appropriate response action, which will vary per application. 

My first use case is to tackle a personal problem of mine (not signing in at the gym) by automatically detecting myself on gym security camera footage and sending an API request. Eventually, this may be extended to other users easily via an opt-in approach, where a user may _e.g._, supply an image of themselves which may then be used to generate their reference embedding and automatically determine classes they attend--even retroactively.

``` mermaid
%%{ init: {"theme": "dark"} }%%
---
title: Application Workflow
---
flowchart
    subgraph Detection
    ce[fa:fa-person Client Enters] --> fd[fa:fa-camera Detect Face]
    fd --> ge[fa:fa-fingerprint Generate Facial Embedding]
    ge --> kc{fa:fa-clock\nUnique Face in\nTime Window?}
    kc --> |Yes| db[(Send Crop to\n Object Storage)] --> pe[fa:fa-envelope Publish Event to\nMessage Bus]
    kc --> |No| p{Privacy?}
    p --> |No| tr[fa:fa-shoe-prints Track Without Label]
    p --> |Yes| d[fa:fa-trash Discard]
    pe --> ar[fa:fa-envelope-open Asynchronous Action\ne.g., Sign-in]
    tr --> wfl{Wait for Label}
    end

    subgraph Labeling
    oi[fa:fa-user-plus Opt-in with User Information\nand Reference Image] --> gs[fa:fa-fingerprint Generate Facial Signature]
    gs --> pl[fa:fa-envelope-circle-check Publish New Label Event]
    pl --> wfl
    wfl --> ra[fa:fa-address-card Retroactively Identify]
    end
    ra --> pe

    classDef sg fill:#133
    class Labeling sg
    class Detection sg
```

## Roadmap
- [x] Limit data generation (_e.g._, do not publish similar embeddings in a given time frame)
- [x] Add publishing to RabbitMQ message bus
- [x] Save detected faces to object storage
- [ ] Benchmark ort (vs tract) with quantized models & switch if appropriate 
- [ ] Create message consumer
- [ ] Move data visualization to a separate binary
- [ ] Generate Docker Compose and/or Helm charts for Postgresql w/ Vector, application, and RabbitMQ
- [ ] Improve startup and shutdown robustness (Nokhwa crate often has issues with initial webcam setup and rude shutdown)
