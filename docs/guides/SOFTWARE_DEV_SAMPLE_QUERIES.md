# Software Development DeepAgent - Sample Queries

This document provides comprehensive sample queries for testing the Software Development DeepAgent across all 9 SDLC phases.

## Table of Contents

1. [Requirements Intelligence](#1-requirements-intelligence)
2. [Architecture Design](#2-architecture-design)
3. [Code Generation](#3-code-generation)
4. [Code Review](#4-code-review)
5. [Testing Automation](#5-testing-automation)
6. [Debugging & Optimization](#6-debugging--optimization)
7. [Security & Compliance](#7-security--compliance)
8. [DevOps Integration](#8-devops-integration)
9. [Documentation](#9-documentation)
10. [Multi-Phase Workflows](#10-multi-phase-workflows)

---

## 1. Requirements Intelligence

### Query 1.1: Analyze Requirements
```
I need to build a user authentication system. Can you analyze these requirements:
- Users should be able to register with email and password
- Support social login (Google, GitHub)
- Two-factor authentication
- Password reset via email
- Session management with JWT tokens
- Role-based access control (Admin, User, Guest)

Please analyze these requirements and identify any ambiguities or missing details.
```

### Query 1.2: Extract User Stories
```
Analyze this product description and extract user stories:

"We need an e-commerce platform where customers can browse products, add items to cart, checkout with payment, and track their orders. Sellers should be able to list products, manage inventory, and view sales reports. The system should support multiple payment methods and send email notifications."

Generate user stories in the format: "As a [role], I want [feature] so that [benefit]"
```

### Query 1.3: Prioritize Requirements
```
Help me prioritize these features for our MVP:
1. User registration and login
2. Product search with filters
3. Shopping cart functionality
4. Payment processing
5. Order tracking
6. Product recommendations
7. Wishlist feature
8. Customer reviews and ratings
9. Email notifications
10. Admin dashboard

Use the Kano model to categorize them as must-be, one-dimensional, attractive, indifferent, or reverse features.
```

### Query 1.4: Generate Acceptance Criteria
```
Generate acceptance criteria for the following user story:

"As a registered user, I want to reset my password so that I can regain access to my account if I forget my password."

Include both functional and non-functional requirements.
```

---

## 2. Architecture Design

### Query 2.1: Design System Architecture
```
Design a microservices architecture for an online food delivery platform with the following requirements:
- Customer mobile app
- Restaurant management portal
- Delivery driver app
- Real-time order tracking
- Payment processing
- High availability (99.9% uptime)
- Support for 10,000 concurrent users
- Scale to 1 million users

Include architecture diagrams, component interactions, and technology recommendations.
```

### Query 2.2: Create REST API Specification
```
Create a REST API specification for a task management system with the following endpoints:
- Create task
- Get task by ID
- Update task status
- Delete task
- List tasks with pagination and filtering
- Assign task to user
- Add comments to task

Generate OpenAPI 3.0 specification with request/response schemas, error codes, and authentication.
```

### Query 2.3: Design Database Schema
```
Design a normalized database schema for a hospital management system that handles:
- Patient records (personal info, medical history)
- Doctor information (specializations, schedules)
- Appointments (booking, status, notes)
- Prescriptions (medications, dosage, instructions)
- Billing and insurance claims

Include entity-relationship diagram, table definitions, indexes, and foreign key relationships.
```

### Query 2.4: Create gRPC API Specification
```
Create a gRPC API specification for a real-time chat service with:
- Send message
- Receive messages (streaming)
- Create chat room
- Join/leave room
- Get user presence status

Generate protobuf definitions with proper message types and service definitions.
```

---

## 3. Code Generation

### Query 3.1: Generate REST API Endpoints
```
Generate Python FastAPI code for a blog API with the following endpoints:
- POST /posts - Create a new blog post
- GET /posts/{id} - Get post by ID
- PUT /posts/{id} - Update post
- DELETE /posts/{id} - Delete post
- GET /posts - List all posts with pagination

Include Pydantic models, database integration (SQLAlchemy), error handling, and authentication middleware.
```

### Query 3.2: Generate Data Models
```
Generate TypeScript data models for an e-commerce system with:
- User (id, email, name, address, created_at)
- Product (id, name, description, price, stock, category)
- Order (id, user_id, items, total_amount, status, created_at)
- OrderItem (order_id, product_id, quantity, price)

Include proper types, interfaces, validation, and relationships.
```

### Query 3.3: Refactor Code
```
Refactor this Python function to improve readability and maintainability:

def process_data(data):
    result = []
    for item in data:
        if item['status'] == 'active' and item['price'] > 0:
            discounted_price = item['price'] * 0.9 if item['category'] == 'sale' else item['price']
            result.append({'id': item['id'], 'name': item['name'], 'price': discounted_price})
    return result

Apply SOLID principles and add proper type hints.
```

### Query 3.4: Generate CRUD Operations
```
Generate a complete CRUD implementation in Java Spring Boot for a Product entity with:
- Fields: id, name, description, price, category, stock
- Repository layer with JPA
- Service layer with business logic
- REST controller with validation
- Exception handling
- Lombok annotations

Include proper HTTP status codes and response DTOs.
```

---

## 4. Code Review

### Query 4.1: Review Code Quality
```
Review this JavaScript function for quality issues:

async function fetchUserData(userId) {
    try {
        const response = await fetch('http://api.example.com/users/' + userId);
        const data = await response.json();
        console.log(data);
        return data;
    } catch (e) {
        console.log('Error:', e);
        return null;
    }
}

Check for: error handling, security issues, best practices, performance, and maintainability.
```

### Query 4.2: Check Security Vulnerabilities
```
Review this Python code for security vulnerabilities:

def login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        session['user_id'] = user[0]
        return redirect('/dashboard')
    else:
        return "Invalid credentials"

Identify all security issues and provide secure alternatives.
```

### Query 4.3: Analyze Code Complexity
```
Analyze the complexity of this algorithm and suggest optimizations:

def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates

Provide time/space complexity analysis and an optimized version.
```

### Query 4.4: Suggest Best Practices
```
Review this React component and suggest improvements based on best practices:

function UserList() {
    const [users, setUsers] = useState([]);

    fetch('/api/users')
        .then(res => res.json())
        .then(data => setUsers(data));

    return (
        <div>
            {users.map((user) => (
                <div>{user.name} - {user.email}</div>
            ))}
        </div>
    );
}

Check for: React best practices, performance issues, error handling, and accessibility.
```

---

## 5. Testing Automation

### Query 5.1: Generate Unit Tests
```
Generate comprehensive unit tests for this Python function:

def calculate_shipping_cost(weight, distance, express=False):
    """Calculate shipping cost based on weight and distance."""
    if weight <= 0 or distance <= 0:
        raise ValueError("Weight and distance must be positive")

    base_cost = weight * 0.5 + distance * 0.1

    if express:
        base_cost *= 1.5

    return round(base_cost, 2)

Use pytest framework and include: happy path, edge cases, error cases, and boundary tests.
```

### Query 5.2: Generate Integration Tests
```
Generate integration tests for a REST API with the following endpoints:
- POST /api/users - Create user
- GET /api/users/{id} - Get user
- PUT /api/users/{id} - Update user
- DELETE /api/users/{id} - Delete user

Test the complete workflow: create → read → update → delete
Include authentication, validation errors, and database state verification.
```

### Query 5.3: Analyze Test Coverage
```
Analyze the test coverage from this pytest-cov report and suggest improvements:

---------- coverage: platform win32, python 3.10.11 -----------
Name                    Stmts   Miss  Cover
-------------------------------------------
src/api/routes.py          45      8    82%
src/models/user.py         30      5    83%
src/services/auth.py       60     15    75%
src/utils/validation.py    25      2    92%
-------------------------------------------
TOTAL                     160     30    81%

Which areas need more test coverage? What types of tests should be added?
```

### Query 5.4: Create Test Plan
```
Create a comprehensive test plan for a mobile banking application with features:
- User authentication (biometric, PIN, password)
- Account balance viewing
- Money transfer between accounts
- Bill payment
- Transaction history
- Push notifications

Include: unit tests, integration tests, E2E tests, security tests, and performance tests.
```

### Query 5.5: Generate Test Data
```
Generate realistic test data for load testing a hotel booking system:
- 100 hotel records with name, location, rating, price range
- 500 user accounts with email, name, phone
- 1000 booking records with check-in/check-out dates, room type, guests

Format as JSON and include edge cases like special characters in names, international phone numbers, and various date ranges.
```

---

## 6. Debugging & Optimization

### Query 6.1: Analyze Error Logs
```
Analyze these error logs and identify the root cause:

2026-01-29 10:15:23 ERROR - Database connection timeout after 30s
2026-01-29 10:15:45 ERROR - Failed to process order #12345: Connection reset by peer
2026-01-29 10:16:10 ERROR - Database connection timeout after 30s
2026-01-29 10:16:32 ERROR - Failed to process order #12346: Connection reset by peer
2026-01-29 10:16:55 ERROR - Max connection pool size reached (50/50)
2026-01-29 10:17:20 ERROR - Failed to process order #12347: Connection reset by peer

What's the root cause? How to fix it? What preventive measures should be implemented?
```

### Query 6.2: Profile Performance
```
This API endpoint is slow (taking 3-5 seconds per request). Profile and optimize:

def get_user_dashboard(user_id):
    user = db.query(User).filter(User.id == user_id).first()
    orders = db.query(Order).filter(Order.user_id == user_id).all()

    for order in orders:
        order.items = db.query(OrderItem).filter(OrderItem.order_id == order.id).all()
        for item in order.items:
            item.product = db.query(Product).filter(Product.id == item.product_id).first()

    recommendations = []
    all_products = db.query(Product).all()
    for product in all_products:
        if product.category in [o.category for o in user.preferences]:
            recommendations.append(product)

    return {
        'user': user,
        'orders': orders,
        'recommendations': recommendations
    }

Identify bottlenecks and provide optimized version.
```

### Query 6.3: Suggest Code Improvements
```
This function works but has quality issues. Suggest improvements:

def process_payment(amount, card_number, cvv, expiry):
    if amount > 0:
        if len(card_number) == 16:
            if len(cvv) == 3:
                if len(expiry) == 5:
                    # Process payment
                    response = requests.post('https://payment.api/charge', json={
                        'amount': amount,
                        'card': card_number,
                        'cvv': cvv,
                        'expiry': expiry
                    })
                    if response.status_code == 200:
                        return True
    return False

Focus on: readability, error handling, security, and maintainability.
```

### Query 6.4: Debug Memory Leak
```
This Python application has a memory leak. Analyze and fix:

class DataProcessor:
    def __init__(self):
        self.cache = {}
        self.callbacks = []

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def process(self, data):
        result = expensive_computation(data)
        self.cache[data] = result

        for callback in self.callbacks:
            callback(result)

        return result

The application runs 24/7 and memory usage grows continuously. What's causing the leak? How to fix it?
```

---

## 7. Security & Compliance

### Query 7.1: Scan for OWASP Top 10 Vulnerabilities
```
Scan this code for OWASP Top 10 vulnerabilities:

@app.route('/search')
def search():
    query = request.args.get('q')
    results = db.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
    return render_template('results.html', results=results, query=query)

@app.route('/upload')
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(f'/uploads/{filename}')
    return 'File uploaded successfully'

@app.route('/admin')
def admin_panel():
    if request.cookies.get('role') == 'admin':
        return render_template('admin.html')
    return 'Access denied'

Identify all vulnerabilities and provide secure alternatives.
```

### Query 7.2: Generate Security Report
```
Generate a comprehensive security report for a web application with:
- Authentication system (JWT-based)
- File upload functionality
- Database with user data
- REST API with public and admin endpoints
- Payment processing integration

Scan for: SQL injection, XSS, CSRF, authentication issues, authorization flaws, sensitive data exposure, and insecure dependencies.
```

### Query 7.3: Check Authentication Security
```
Review this authentication implementation for security issues:

def authenticate_user(username, password):
    user = User.query.filter_by(username=username).first()

    if user and user.password == password:
        token = jwt.encode({'user_id': user.id}, 'secret-key')
        return {'token': token, 'user': user.to_dict()}

    return None

def verify_token(token):
    try:
        data = jwt.decode(token, 'secret-key', algorithms=['HS256'])
        return data['user_id']
    except:
        return None

Identify security vulnerabilities and provide secure implementation.
```

### Query 7.4: Analyze Dependency Vulnerabilities
```
Analyze these Python dependencies for known vulnerabilities:

flask==2.0.1
requests==2.25.1
pillow==8.3.1
django==3.1.12
jinja2==2.11.3
pyyaml==5.3.1
cryptography==3.3.2

Check against CVE database and recommend secure versions or alternatives.
```

---

## 8. DevOps Integration

### Query 8.1: Generate Dockerfile
```
Generate a production-ready Dockerfile for a Node.js Express application with:
- Node.js 18 LTS
- Multi-stage build for smaller image size
- Security best practices (non-root user, minimal base image)
- Health check endpoint
- Environment variable configuration
- npm install with cache optimization

The application structure:
- src/ (application code)
- package.json, package-lock.json
- .env.example (environment variables)
- Entry point: src/server.js
```

### Query 8.2: Create CI/CD Pipeline
```
Create a GitHub Actions CI/CD pipeline for a Python FastAPI application with:
- Run tests with pytest
- Check code quality with ruff
- Build Docker image
- Push to Docker Hub
- Deploy to Azure App Service (production)
- Send Slack notification on success/failure

Include separate workflows for: pull requests (test only) and main branch (test + deploy).
```

### Query 8.3: Generate Kubernetes Configuration
```
Generate Kubernetes manifests for a microservices application with:
- Frontend service (React app, 3 replicas, port 80)
- Backend API (Node.js, 5 replicas, port 3000, HPA based on CPU 70%)
- Database (PostgreSQL, 1 replica, persistent volume)
- Redis cache (1 replica)

Include: Deployments, Services, ConfigMaps, Secrets, Ingress, and HorizontalPodAutoscaler.
```

### Query 8.4: Create Monitoring Configuration
```
Create monitoring and alerting configuration for a production web application:
- Application: Python FastAPI
- Deployment: Kubernetes on Azure
- Requirements:
  - Monitor API response times
  - Track error rates
  - Database connection pool metrics
  - Memory and CPU usage
  - Alert on: 5xx errors > 1%, p99 latency > 2s, pod restarts

Generate configurations for: Prometheus, Grafana dashboards, and alert rules.
```

---

## 9. Documentation

### Query 9.1: Generate API Documentation
```
Generate comprehensive API documentation for this FastAPI application:

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Task(BaseModel):
    id: int
    title: str
    description: str
    status: str
    priority: int

@app.post("/tasks", response_model=Task)
async def create_task(task: Task):
    # Create task logic
    return task

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: int):
    # Get task logic
    pass

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, task: Task):
    # Update task logic
    pass

Include: endpoint descriptions, request/response examples, error codes, authentication requirements, and usage examples in multiple languages (curl, Python, JavaScript).
```

### Query 9.2: Create User Guide
```
Create a comprehensive user guide for a REST API client library with:
- Installation instructions (pip, npm, maven)
- Quick start guide
- Authentication setup
- Common use cases with code examples
- Error handling
- Rate limiting and best practices
- Troubleshooting section

Target audience: developers integrating with our API.
```

### Query 9.3: Generate Architecture Documentation
```
Generate architecture documentation for a microservices-based e-commerce platform with:
- System overview
- Architecture diagram (C4 model - Context, Container, Component, Code)
- Service descriptions (frontend, product service, order service, payment service, notification service)
- Data flow diagrams
- Technology stack
- Deployment architecture
- Security architecture
- Scalability considerations

Include Mermaid diagrams for visualizations.
```

### Query 9.4: Create Code Comments
```
Add comprehensive documentation comments to this TypeScript class:

class UserService {
    constructor(private db: Database, private cache: Cache) {}

    async createUser(userData: CreateUserDTO): Promise<User> {
        const user = await this.db.users.create(userData);
        await this.cache.set(`user:${user.id}`, user, 3600);
        return user;
    }

    async getUser(userId: string): Promise<User | null> {
        const cached = await this.cache.get(`user:${userId}`);
        if (cached) return cached;

        const user = await this.db.users.findById(userId);
        if (user) {
            await this.cache.set(`user:${userId}`, user, 3600);
        }
        return user;
    }
}

Use TSDoc format with proper tags for parameters, returns, and examples.
```

---

## 10. Multi-Phase Workflows

### Query 10.1: End-to-End Feature Development
```
I need to build a complete user authentication feature with email verification. Please help me:

1. Analyze requirements and extract user stories
2. Design the database schema and API endpoints
3. Generate backend code (Python FastAPI)
4. Generate unit and integration tests
5. Review the code for security vulnerabilities
6. Create API documentation
7. Generate Dockerfile and CI/CD pipeline

Requirements:
- Email/password registration
- Email verification with token
- Login with JWT authentication
- Password reset via email
- Session management
- Rate limiting on auth endpoints
```

### Query 10.2: Legacy Code Modernization
```
Help me modernize this legacy codebase:

Current state:
- Monolithic PHP application (10 years old)
- No tests
- MySQL database with no ORM
- Manual deployment
- Poor documentation

Target state:
- Microservices architecture
- Python FastAPI or Node.js
- PostgreSQL with ORM
- Comprehensive tests (80%+ coverage)
- Docker + Kubernetes deployment
- CI/CD pipeline
- OpenAPI documentation

Please provide:
1. Analysis of current code and architecture
2. Migration strategy and architecture design
3. Code refactoring approach
4. Testing strategy
5. Deployment plan
6. Risk assessment and mitigation
```

### Query 10.3: Performance Optimization Sprint
```
Our application has performance issues. Help me optimize:

Current issues:
- API response time: 3-5 seconds (target: <500ms)
- Database queries: N+1 problem
- Memory usage: 2GB per instance (target: <500MB)
- No caching layer

Steps needed:
1. Profile the application and identify bottlenecks
2. Review and optimize database queries
3. Implement caching strategy (Redis)
4. Optimize code algorithms
5. Add performance tests
6. Set up monitoring and alerting
7. Document optimization changes

Target metrics:
- P95 latency < 500ms
- Throughput > 1000 req/s
- Memory usage < 500MB
- CPU usage < 50% at peak load
```

### Query 10.4: Security Audit and Remediation
```
Conduct a comprehensive security audit of our web application and fix all issues:

Application details:
- Node.js Express backend
- React frontend
- MongoDB database
- User authentication with JWT
- File upload functionality
- Payment integration (Stripe)
- Deployed on AWS

Audit requirements:
1. Scan for OWASP Top 10 vulnerabilities
2. Review authentication and authorization
3. Check for sensitive data exposure
4. Analyze dependency vulnerabilities
5. Review API security (rate limiting, input validation)
6. Check deployment security (HTTPS, CORS, security headers)
7. Generate security report with remediation steps
8. Provide secure code examples for all issues found

Compliance: GDPR, PCI-DSS (for payment data)
```

---

## Testing Tips

### UI Testing
1. Navigate to: http://localhost:8000/chat
2. Select "software_development" from the agent dropdown
3. Copy and paste queries from this document
4. Observe the agent's responses and tool invocations
5. Verify that appropriate tools are called for each query

### REST API Testing
Use curl or Postman to test the REST endpoint:

```bash
curl -X POST http://localhost:8000/api/deepagent/software_development/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate unit tests for a Python function that validates email addresses",
    "session_id": "test-session-123"
  }'
```

### Expected Behavior
- The agent should route queries to appropriate subagents
- Tool calls should be visible in responses
- Multi-step workflows should show intermediate results
- Session context should be maintained across messages
- Error handling should be graceful

### Success Criteria
- ✅ Agent loads without errors (14 agents total)
- ✅ All 54 tools are available
- ✅ Queries are routed to correct subagents
- ✅ Tool invocations return valid JSON
- ✅ Multi-turn conversations maintain context
- ✅ Code generation includes proper syntax
- ✅ Security scanning identifies vulnerabilities
- ✅ Documentation is well-formatted

---

## Troubleshooting

### Agent Not Found
If you see "Error: DeepAgent software_development not found":
1. Restart the server (see [RESTART_SERVER.md](RESTART_SERVER.md))
2. Check server logs for loading errors
3. Verify `.env` file has `AZURE_OPENAI_API_KEY` configured

### Tools Not Working
If tool invocations fail:
1. Check tool parameters match expected types
2. Verify session_id is consistent
3. Check server logs for tool errors

### Slow Responses
If responses are slow:
1. Complex queries may take 30-60 seconds
2. Multi-phase workflows involve multiple tool calls
3. Check Azure OpenAI API rate limits

---

## Additional Resources

- [Architecture Documentation](docs/ENTERPRISE_ARCHITECTURE.md)
- [Server Restart Procedure](RESTART_SERVER.md)
- [Software Development DeepAgent Implementation](libs/azure-ai/langchain_azure_ai/wrappers/deep_agents.py)
- [Tool Implementations](libs/azure-ai/langchain_azure_ai/wrappers/software_dev_tools/)
