# Software Engineering & Cloud Roadmap

## 1. Engineering Disciplines & Skillsets

### Frontend Engineering
- **Core Languages:** Modern JavaScript (ES6+ features), TypeScript for type safety and clean refactoring.
- **Frameworks:** React.js, Next.js (App Router, Server Components, SSR/SSG), Vue.js, Svelte.
- **Styling & Design Systems:** Tailwind CSS, CSS Modules, Styled Components, Responsive Design, Accessibility (WCAG 2.1).
- **State Management & Caching:** Redux Toolkit, Zustand, TanStack Query (React Query) for server state.
- **Performance:** Core Web Vitals (LCP, FID/INP, CLS), code splitting, lazy loading, image optimization.

### Backend Engineering
- **Core Stacks:** Python (FastAPI, Django), Node.js (Express, NestJS), Go, or Java/Spring Boot.
- **API Architecture:** RESTful best practices, GraphQL schemas, gRPC for internal service communication, WebSocket for real-time bidirectional data.
- **Databases:**
  - Relational: PostgreSQL, MySQL (indexing, normalization, ACID guarantees, query plans).
  - NoSQL & Caching: MongoDB, Redis (caching, pub/sub, rate limiting).
- **Asynchronous Task Processing:** Celery, Redis Queue, RabbitMQ, Apache Kafka for event-driven architectures.

### DevOps & Cloud Infrastructure
- **Cloud Providers:** AWS (EC2, S3, RDS, Lambda, ECS/EKS), Google Cloud, Azure.
- **Containerization & Orchestration:** Docker, Docker Compose, Kubernetes fundamentals (Pods, Deployments, Services, Ingress).
- **Infrastructure as Code (IaC):** Terraform, AWS CloudFormation.
- **CI/CD Automation:** GitHub Actions, GitLab CI, automated test runners, artifact registry.
- **Monitoring & Observability:** Prometheus, Grafana, OpenTelemetry, ELK stack (Elasticsearch, Logstash, Kibana).

## 2. System Design Best Practices
- **Scalability Principles:** Horizontal vs vertical scaling, load balancers, database read-replicas, caching tiers.
- **Resilience:** Circuit breakers, exponential backoff retries, rate limiting, health check probes.
- **Security:** OAuth 2.0 / JWT token authentication, CORS configuration, SQL injection prevention, secrets management.

## 3. Recommended Portfolio Showcases
- **Distributed Task Scheduler:** A microservices backend handling asynchronous jobs with Redis queue and live websocket updates.
- **Full-Stack SaaS Platform:** Complete application featuring user auth, Stripe billing integration, RBAC, and responsive dashboard.
