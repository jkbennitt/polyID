# GitHub Projects Templates and Automation Files

This document contains all the template files, scripts, and automation configurations needed to implement the GitHub Projects setup for polyID.

## Table of Contents

1. [Workflow Automation Files](#workflow-automation-files)
2. [Project Setup Scripts](#project-setup-scripts)
3. [Configuration Templates](#configuration-templates)
4. [Integration Examples](#integration-examples)

## Workflow Automation Files

### 1. Project Automation Workflow

Create `.github/workflows/project_automation.yml`:

```yaml
name: GitHub Projects Automation

on:
  issues:
    types: [opened, labeled, closed]
  pull_request:
    types: [opened, closed, ready_for_review]
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  route_issues:
    runs-on: ubuntu-latest
    if: github.event_name == 'issues'
    steps:
      - name: Route Research Issues
        if: |
          contains(github.event.issue.labels.*.name, 'research') || 
          contains(github.event.issue.labels.*.name, 'enhancement') ||
          contains(github.event.issue.labels.*.name, 'research:modeling') ||
          contains(github.event.issue.labels.*.name, 'research:validation')
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/1
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}

      - name: Route Development Issues
        if: |
          contains(github.event.issue.labels.*.name, 'bug') ||
          contains(github.event.issue.labels.*.name, 'documentation') ||
          contains(github.event.issue.labels.*.name, 'infrastructure') ||
          contains(github.event.issue.labels.*.name, 'performance')
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/2
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}

      - name: Route Release Issues
        if: |
          contains(github.event.issue.labels.*.name, 'release') ||
          contains(github.event.issue.labels.*.name, 'version') ||
          contains(github.event.issue.labels.*.name, 'deployment')
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/3
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}

  handle_pull_requests:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Add PR to Development Board
        if: github.event.action == 'opened' || github.event.action == 'ready_for_review'
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/2
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}

      - name: Handle PR Merge
        if: github.event.pull_request.merged == true
        uses: actions/github-script@v6
        with:
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
          script: |
            // Update project item status to "Done" when PR is merged
            console.log('PR merged - updating project status')
            // Additional automation logic can be added here

  handle_ci_failures:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_run' && github.event.workflow_run.conclusion == 'failure'
    steps:
      - name: Create CI Failure Issue
        uses: actions/github-script@v6
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `CI Failure: ${{ github.event.workflow_run.name }} - Run #${{ github.event.workflow_run.run_number }}`,
              body: `**Automated CI Failure Report**
              
              **Details:**
              - Workflow: ${{ github.event.workflow_run.name }}
              - Run ID: ${{ github.event.workflow_run.id }}
              - Commit: ${{ github.event.workflow_run.head_commit.id }}
              - Branch: ${{ github.event.workflow_run.head_branch }}
              - Conclusion: ${{ github.event.workflow_run.conclusion }}
              
              **Logs:** [View Run](${{ github.event.workflow_run.html_url }})
              
              **Next Steps:**
              - [ ] Review build logs
              - [ ] Identify root cause
              - [ ] Implement fix
              - [ ] Verify fix with tests
              
              This issue was automatically created by the project automation system.`,
              labels: ['bug', 'ci-failure', 'priority:high', 'automated']
            });
            
            console.log('Created CI failure issue:', issue.data.number);

      - name: Add CI Failure to Development Board
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/2
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
```

### 2. Performance Monitoring Integration

Create `.github/workflows/performance_project_integration.yml`:

```yaml
name: Performance Issue Project Integration

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  check_performance_metrics:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Performance Check
        id: perf_check
        run: |
          python -c "
          from polyid.performance_monitor import get_performance_monitor
          import json
          
          monitor = get_performance_monitor()
          metrics = monitor.get_current_metrics()
          
          # Check for performance issues
          issues = []
          if metrics.get('memory_usage', 0) > 80:
              issues.append('High memory usage detected')
          if metrics.get('prediction_time', 0) > 5.0:
              issues.append('Slow prediction times detected')
          if metrics.get('cache_hit_rate', 100) < 60:
              issues.append('Low cache hit rate detected')
              
          print(f'::set-output name=issues::{json.dumps(issues)}')
          print(f'::set-output name=metrics::{json.dumps(metrics)}')
          "

      - name: Create Performance Issues
        if: steps.perf_check.outputs.issues != '[]'
        uses: actions/github-script@v6
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const issues = JSON.parse('${{ steps.perf_check.outputs.issues }}');
            const metrics = JSON.parse('${{ steps.perf_check.outputs.metrics }}');
            
            for (const issue of issues) {
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: `Performance Alert: ${issue}`,
                body: `**Automated Performance Alert**
                
                **Issue:** ${issue}
                
                **Current Metrics:**
                - Memory Usage: ${metrics.memory_usage}%
                - Prediction Time: ${metrics.prediction_time}s
                - Cache Hit Rate: ${metrics.cache_hit_rate}%
                
                **Investigation Steps:**
                - [ ] Review performance logs
                - [ ] Check system resources
                - [ ] Identify bottlenecks
                - [ ] Implement optimizations
                - [ ] Verify improvements
                
                This issue was automatically created by performance monitoring.`,
                labels: ['performance', 'automated', 'priority:medium']
              });
            }
```

### 3. Release Automation Integration

Create `.github/workflows/release_project_integration.yml`:

```yaml
name: Release Project Integration

on:
  release:
    types: [published, prereleased]
  push:
    tags:
      - 'v*'

jobs:
  update_release_board:
    runs-on: ubuntu-latest
    steps:
      - name: Extract version info
        id: version
        run: |
          if [[ "${GITHUB_REF}" =~ ^refs/tags/v(.*)$ ]]; then
            echo "::set-output name=version::${BASH_REMATCH[1]}"
            if [[ "${BASH_REMATCH[1]}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
              echo "::set-output name=type::release"
            else
              echo "::set-output name=type::prerelease"
            fi
          fi

      - name: Create Release Completed Issue
        uses: actions/github-script@v6
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const version = '${{ steps.version.outputs.version }}';
            const type = '${{ steps.version.outputs.type }}';
            
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Post-Release Tasks: v${version}`,
              body: `**Release Completed: v${version}**
              
              **Release Type:** ${type}
              **Release Date:** ${new Date().toISOString().split('T')[0]}
              
              **Post-Release Checklist:**
              - [ ] Monitor deployment health
              - [ ] Check PyPI publication
              - [ ] Verify HF Spaces deployment
              - [ ] Update documentation
              - [ ] Monitor for user feedback
              - [ ] Check performance metrics
              - [ ] Update PaleoBond-PCP compatibility
              
              **Monitoring Period:** 7 days
              
              This issue will track post-release activities and can be closed after the monitoring period.`,
              labels: ['release', 'post-release', 'monitoring', `version:${version}`]
            });

      - name: Add to Release Management Board
        uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/${{ github.repository_owner }}/projects/3
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
```

## Project Setup Scripts

### 1. Main Setup Script

Create `scripts/setup_github_projects.sh`:

```bash
#!/bin/bash
# Setup script for polyID GitHub Projects

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up polyID GitHub Projects...${NC}"

# Check prerequisites
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed${NC}"
    echo "Please install from: https://cli.github.com/"
    exit 1
fi

# Authenticate with GitHub CLI
echo -e "${YELLOW}Checking GitHub CLI authentication...${NC}"
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}Please authenticate with GitHub CLI:${NC}"
    gh auth login
fi

# Get repository information
REPO_OWNER=$(gh repo view --json owner --jq .owner.login)
REPO_NAME=$(gh repo view --json name --jq .name)

echo -e "${GREEN}Repository: ${REPO_OWNER}/${REPO_NAME}${NC}"

# Create projects
echo -e "${YELLOW}Creating GitHub Projects...${NC}"

# Research Board
echo "Creating Research Board..."
RESEARCH_PROJECT=$(gh project create --owner "$REPO_OWNER" \
    --title "polyID Research Board" \
    --body "Scientific research, model development, and validation tracking for polyID" \
    --format json | jq -r .number)

# Development Board
echo "Creating Development Board..."
DEV_PROJECT=$(gh project create --owner "$REPO_OWNER" \
    --title "polyID Development Board" \
    --body "Feature development, bug fixes, and infrastructure management for polyID" \
    --format json | jq -r .number)

# Release Management Board
echo "Creating Release Management Board..."
RELEASE_PROJECT=$(gh project create --owner "$REPO_OWNER" \
    --title "polyID Release Management" \
    --body "Version planning, release coordination, and post-release activities for polyID" \
    --format json | jq -r .number)

echo -e "${GREEN}Projects created successfully!${NC}"
echo "Research Board: Project #${RESEARCH_PROJECT}"
echo "Development Board: Project #${DEV_PROJECT}"
echo "Release Management: Project #${RELEASE_PROJECT}"

# Update project URLs in workflow files
echo -e "${YELLOW}Updating workflow files with project URLs...${NC}"

# Create temporary directory for project configuration
mkdir -p .github/project-config

# Save project numbers for later reference
cat > .github/project-config/project-numbers.json <<EOF
{
  "research_board": ${RESEARCH_PROJECT},
  "development_board": ${DEV_PROJECT},
  "release_management": ${RELEASE_PROJECT},
  "owner": "${REPO_OWNER}"
}
EOF

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Configure custom fields using the GitHub Projects UI"
echo "2. Set up project views and columns"
echo "3. Configure automation rules"
echo "4. Import existing issues and PRs"
echo "5. Update team permissions"
echo ""
echo "Refer to .github/GITHUB_PROJECTS_SETUP_GUIDE.md for detailed instructions."
```

### 2. Field Configuration Script

Create `scripts/configure_project_fields.py`:

```python
#!/usr/bin/env python3
"""
Script to configure custom fields for polyID GitHub Projects
"""

import json
import subprocess
import sys
from typing import Dict, List

def run_gh_command(cmd: List[str]) -> dict:
    """Run a GitHub CLI command and return JSON result"""
    try:
        result = subprocess.run(['gh'] + cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(['gh'] + cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def create_select_field(project_number: str, field_name: str, options: List[str]) -> None:
    """Create a single select field in a project"""
    print(f"Creating field '{field_name}' in project {project_number}")
    
    # Note: This is pseudocode as the actual gh CLI commands for custom fields
    # may vary. This represents the intended configuration.
    cmd = [
        'project', 'field-create', str(project_number),
        '--name', field_name,
        '--type', 'single_select'
    ]
    
    for option in options:
        cmd.extend(['--option', option])
    
    try:
        subprocess.run(['gh'] + cmd, check=True)
        print(f"✓ Created field '{field_name}'")
    except subprocess.CalledProcessError:
        print(f"✗ Failed to create field '{field_name}'")

def setup_research_board_fields(project_number: str) -> None:
    """Configure custom fields for Research Board"""
    print(f"\nConfiguring Research Board (Project #{project_number}) fields...")
    
    fields_config = {
        "Research Phase": [
            "Literature Review",
            "Data Collection", 
            "Model Development",
            "Validation",
            "Analysis",
            "Publication"
        ],
        "Research Priority": [
            "Critical",
            "High", 
            "Medium",
            "Low"
        ],
        "Complexity": [
            "Simple (1-3 days)",
            "Moderate (1-2 weeks)", 
            "Complex (2-4 weeks)",
            "Research Project (1-3 months)"
        ],
        "Model Type": [
            "Neural Network",
            "Feature Engineering",
            "Preprocessing", 
            "Validation Method",
            "Performance Optimization"
        ],
        "Scientific Impact": [
            "High (Novel discovery)",
            "Medium (Improvement)",
            "Low (Maintenance)"
        ],
        "Validation Status": [
            "Not Started",
            "In Progress",
            "Peer Review", 
            "Validated",
            "Published"
        ]
    }
    
    for field_name, options in fields_config.items():
        create_select_field(project_number, field_name, options)

def setup_development_board_fields(project_number: str) -> None:
    """Configure custom fields for Development Board"""
    print(f"\nConfiguring Development Board (Project #{project_number}) fields...")
    
    fields_config = {
        "Development Type": [
            "Feature",
            "Bug Fix",
            "Infrastructure", 
            "Documentation",
            "Performance",
            "Security"
        ],
        "Sprint": [
            "Backlog",
            "Current Sprint",
            "Next Sprint", 
            "Future"
        ],
        "Technical Complexity": [
            "Low (< 1 day)",
            "Medium (1-3 days)",
            "High (3-7 days)", 
            "Epic (> 1 week)"
        ],
        "Component": [
            "Core ML (polyid/)",
            "Interface (app.py)",
            "Performance (optimizations)",
            "Testing (tests/)",
            "Documentation (docs/)", 
            "Infrastructure (.github/)",
            "Deployment (Docker/HF)"
        ],
        "Review Status": [
            "Draft",
            "Ready for Review",
            "In Review",
            "Approved", 
            "Merged"
        ],
        "Testing Required": [
            "Unit Tests",
            "Integration Tests", 
            "Performance Tests",
            "Manual Testing",
            "No Testing"
        ]
    }
    
    for field_name, options in fields_config.items():
        create_select_field(project_number, field_name, options)

def setup_release_board_fields(project_number: str) -> None:
    """Configure custom fields for Release Management Board"""
    print(f"\nConfiguring Release Management Board (Project #{project_number}) fields...")
    
    fields_config = {
        "Release Version": [
            "v1.1.0 (Next Patch)",
            "v1.2.0 (Next Minor)", 
            "v2.0.0 (Next Major)",
            "Future"
        ],
        "Release Type": [
            "Patch (Bug fixes)",
            "Minor (New features)",
            "Major (Breaking changes)"
        ],
        "Release Priority": [
            "Critical (Hotfix)",
            "High (Planned)",
            "Medium (Enhancement)", 
            "Low (Nice-to-have)"
        ],
        "Deployment Target": [
            "PyPI",
            "TestPyPI",
            "GitHub Release",
            "HF Spaces", 
            "Docker Hub"
        ],
        "Release Status": [
            "Planning",
            "Development",
            "Testing",
            "Release Candidate", 
            "Released",
            "Post-Release"
        ]
    }
    
    for field_name, options in fields_config.items():
        create_select_field(project_number, field_name, options)

def main():
    """Main configuration function"""
    print("Configuring polyID GitHub Projects custom fields...")
    
    # Load project numbers
    try:
        with open('.github/project-config/project-numbers.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Project configuration not found. Run setup_github_projects.sh first.")
        sys.exit(1)
    
    # Configure each board
    setup_research_board_fields(config['research_board'])
    setup_development_board_fields(config['development_board'])
    setup_release_board_fields(config['release_management'])
    
    print("\n✅ Custom fields configuration complete!")
    print("Next steps:")
    print("1. Set up project views and column configurations")
    print("2. Configure automation rules")
    print("3. Import existing issues and PRs")

if __name__ == "__main__":
    main()
```

## Configuration Templates

### 1. Project Views Configuration

Create `.github/project-templates/views-config.json`:

```json
{
  "research_board_views": {
    "kanban_board": {
      "name": "Research Workflow",
      "type": "board",
      "columns": [
        "📚 Literature Review",
        "🔬 Hypothesis/Design", 
        "🧪 Development",
        "📊 Validation",
        "✅ Validated",
        "📝 Documentation"
      ]
    },
    "research_table": {
      "name": "Research Overview",
      "type": "table",
      "fields": [
        "Title",
        "Research Phase", 
        "Scientific Impact",
        "Complexity",
        "Assignee",
        "Status"
      ]
    },
    "timeline": {
      "name": "Research Timeline", 
      "type": "timeline",
      "start_date_field": "Created",
      "target_date_field": "Due Date"
    }
  },
  "development_board_views": {
    "sprint_board": {
      "name": "Development Sprint",
      "type": "board", 
      "columns": [
        "📋 Backlog",
        "🎯 Ready",
        "🔨 In Progress", 
        "🔍 Code Review",
        "🧪 Testing",
        "✅ Done"
      ]
    },
    "development_table": {
      "name": "Feature Tracking",
      "type": "table",
      "fields": [
        "Title",
        "Development Type",
        "Component", 
        "Technical Complexity",
        "Sprint",
        "Assignee"
      ]
    },
    "bug_triage": {
      "name": "Bug Triage",
      "type": "table",
      "filters": ["Development Type:Bug Fix"],
      "fields": [
        "Title", 
        "Priority",
        "Component",
        "Review Status",
        "Assignee"
      ]
    }
  },
  "release_board_views": {
    "release_timeline": {
      "name": "Release Roadmap",
      "type": "timeline",
      "start_date_field": "Start Date",
      "target_date_field": "Target Date" 
    },
    "version_table": {
      "name": "Version Planning",
      "type": "table",
      "fields": [
        "Title",
        "Release Version",
        "Release Type",
        "Release Priority", 
        "Deployment Target",
        "Release Status"
      ]
    },
    "release_board": {
      "name": "Current Release",
      "type": "board",
      "columns": [
        "📋 Planning",
        "🔨 Development", 
        "🧪 Testing",
        "🚀 Ready to Release",
        "✅ Released",
        "📊 Post-Release"
      ]
    }
  }
}
```

### 2. Automation Rules Template

Create `.github/project-templates/automation-rules.json`:

```json
{
  "rules": [
    {
      "name": "Auto-route bug reports",
      "trigger": "item_added",
      "conditions": [
        {
          "field": "labels",
          "operator": "contains", 
          "value": "bug"
        }
      ],
      "actions": [
        {
          "type": "set_field",
          "field": "Status",
          "value": "Backlog"
        },
        {
          "type": "set_field", 
          "field": "Development Type",
          "value": "Bug Fix"
        }
      ]
    },
    {
      "name": "Auto-route research issues",
      "trigger": "item_added",
      "conditions": [
        {
          "field": "labels",
          "operator": "contains",
          "value": "research"
        }
      ],
      "actions": [
        {
          "type": "set_field",
          "field": "Research Phase", 
          "value": "Literature Review"
        }
      ]
    },
    {
      "name": "Move to review on PR ready",
      "trigger": "pull_request_ready_for_review",
      "actions": [
        {
          "type": "set_field",
          "field": "Status",
          "value": "Code Review"
        },
        {
          "type": "set_field",
          "field": "Review Status", 
          "value": "Ready for Review"
        }
      ]
    },
    {
      "name": "Complete on PR merge",
      "trigger": "pull_request_merged",
      "actions": [
        {
          "type": "set_field",
          "field": "Status", 
          "value": "Done"
        },
        {
          "type": "set_field",
          "field": "Review Status",
          "value": "Merged"
        }
      ]
    },
    {
      "name": "Archive old completed items",
      "trigger": "scheduled",
      "schedule": "weekly",
      "conditions": [
        {
          "field": "Status",
          "operator": "equals",
          "value": "Done"
        },
        {
          "field": "Updated",
          "operator": "older_than",
          "value": "30 days"
        }
      ],
      "actions": [
        {
          "type": "archive_item"
        }
      ]
    }
  ]
}
```

## Integration Examples

### 1. Performance Monitoring Integration

Example Python code for integrating performance monitoring with projects:

```python
# polyid/project_integration.py
"""
Integration between polyID performance monitoring and GitHub Projects
"""

import os
import json
import requests
from typing import Dict, List, Optional
from .performance_monitor import get_performance_monitor

class GitHubProjectsIntegration:
    """Integration with GitHub Projects for performance tracking"""
    
    def __init__(self, github_token: str, project_number: str):
        self.github_token = github_token
        self.project_number = project_number
        self.base_url = "https://api.github.com/graphql"
        
    def create_performance_issue(self, issue_type: str, metrics: Dict) -> Optional[str]:
        """Create a performance issue in GitHub Projects"""
        
        # Create issue via GitHub API
        issue_data = {
            "title": f"Performance Alert: {issue_type}",
            "body": self._generate_performance_issue_body(issue_type, metrics),
            "labels": ["performance", "automated", "priority:medium"]
        }
        
        # This would integrate with GitHub API to create the issue
        # and add it to the appropriate project board
        return self._create_github_issue(issue_data)
    
    def _generate_performance_issue_body(self, issue_type: str, metrics: Dict) -> str:
        """Generate issue body for performance alerts"""
        return f"""**Automated Performance Alert**

**Issue Type:** {issue_type}

**Current Metrics:**
- Memory Usage: {metrics.get('memory_usage', 'N/A')}%
- Prediction Time: {metrics.get('prediction_time', 'N/A')}s  
- Cache Hit Rate: {metrics.get('cache_hit_rate', 'N/A')}%
- Error Rate: {metrics.get('error_rate', 'N/A')}%

**Performance Thresholds:**
- Memory Usage: >80% (Current: {metrics.get('memory_usage', 'N/A')}%)
- Prediction Time: >5.0s (Current: {metrics.get('prediction_time', 'N/A')}s)
- Cache Hit Rate: <60% (Current: {metrics.get('cache_hit_rate', 'N/A')}%)

**Investigation Checklist:**
- [ ] Review performance logs
- [ ] Check system resources  
- [ ] Identify bottlenecks
- [ ] Implement optimizations
- [ ] Verify improvements
- [ ] Update monitoring thresholds if needed

**Monitoring Period:** This issue will auto-resolve if metrics return to normal ranges.

*This issue was automatically created by the performance monitoring system.*
"""

    def _create_github_issue(self, issue_data: Dict) -> Optional[str]:
        """Create issue via GitHub API"""
        # Implementation would use GitHub API to create issue
        # and add to project board
        pass

# Performance monitoring integration
def check_and_report_performance_issues():
    """Check performance metrics and create GitHub issues if needed"""
    
    monitor = get_performance_monitor()
    metrics = monitor.get_current_metrics()
    
    # Load configuration
    github_token = os.environ.get('GITHUB_TOKEN')
    project_number = os.environ.get('DEVELOPMENT_PROJECT_NUMBER')
    
    if not github_token or not project_number:
        return
        
    integration = GitHubProjectsIntegration(github_token, project_number)
    
    # Check thresholds
    issues_to_create = []
    
    if metrics.get('memory_usage', 0) > 80:
        issues_to_create.append(('High Memory Usage', metrics))
        
    if metrics.get('prediction_time', 0) > 5.0:
        issues_to_create.append(('Slow Prediction Times', metrics))
        
    if metrics.get('cache_hit_rate', 100) < 60:
        issues_to_create.append(('Low Cache Hit Rate', metrics))
    
    # Create issues for performance problems
    for issue_type, issue_metrics in issues_to_create:
        integration.create_performance_issue(issue_type, issue_metrics)
```

### 2. Research Workflow Integration

Example for integrating research activities:

```python
# polyid/research_integration.py
"""
Integration for research workflow tracking
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ResearchTask:
    """Research task representation"""
    title: str
    phase: str  # Literature Review, Development, Validation, etc.
    complexity: str
    scientific_impact: str
    model_type: Optional[str] = None
    validation_status: str = "Not Started"
    
class ResearchWorkflowTracker:
    """Track research workflow progress"""
    
    def __init__(self):
        self.active_tasks: List[ResearchTask] = []
    
    def start_literature_review(self, topic: str) -> ResearchTask:
        """Start a literature review task"""
        task = ResearchTask(
            title=f"Literature Review: {topic}",
            phase="Literature Review",
            complexity="Simple (1-3 days)",
            scientific_impact="Medium (Improvement)"
        )
        self.active_tasks.append(task)
        return task
    
    def start_model_development(self, model_name: str, model_type: str) -> ResearchTask:
        """Start model development task"""
        task = ResearchTask(
            title=f"Develop {model_name} Model",
            phase="Model Development", 
            complexity="Complex (2-4 weeks)",
            scientific_impact="High (Novel discovery)",
            model_type=model_type
        )
        self.active_tasks.append(task)
        return task
    
    def start_validation(self, model_name: str) -> ResearchTask:
        """Start model validation"""
        task = ResearchTask(
            title=f"Validate {model_name} Model",
            phase="Validation",
            complexity="Moderate (1-2 weeks)", 
            scientific_impact="High (Novel discovery)",
            validation_status="In Progress"
        )
        self.active_tasks.append(task)
        return task

# Example usage in research workflow
def example_research_workflow():
    """Example of how research tasks would be tracked"""
    
    tracker = ResearchWorkflowTracker()
    
    # Start literature review
    lit_review = tracker.start_literature_review("Graph Neural Networks for Polymers")
    
    # Start model development 
    model_dev = tracker.start_model_development("Enhanced GNN", "Neural Network")
    
    # Start validation
    validation = tracker.start_validation("Enhanced GNN")
    
    return tracker
```

### 3. CI/CD Integration Examples

Example integrations with existing CI/CD workflows:

```yaml
# Additional steps to add to .github/workflows/ci.yml

# Add to the test job
- name: Update Development Board on Test Results
  if: always()
  uses: actions/github-script@v6
  with:
    github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
    script: |
      const success = '${{ job.status }}' === 'success';
      const testResults = {
        status: '${{ job.status }}',
        python_version: '${{ matrix.python-version }}',
        timestamp: new Date().toISOString()
      };
      
      // Update project items related to current PR or commit
      // Implementation would query project items and update status
      console.log('Test results:', testResults);

# Add to the build job  
- name: Update Release Board on Build Success
  if: success() && startsWith(github.ref, 'refs/tags/')
  uses: actions/github-script@v6
  with:
    github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
    script: |
      // Extract version from tag
      const tag = context.ref.replace('refs/tags/', '');
      console.log('Build successful for tag:', tag);
      
      // Update release board items
      // Mark build as complete, ready for testing phase
```

### 4. PaleoBond-PCP Integration

Since polyID is a microservice for PaleoBond-PCP:

```python
# polyid/paleobond_integration.py
"""
Integration helpers for PaleoBond-PCP project coordination
"""

from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class PaleoBondCompatibilityIssue:
    """Track compatibility issues with main project"""
    api_version: str
    breaking_change: bool
    impact_level: str  # Low, Medium, High, Critical
    affected_components: List[str]
    mitigation_plan: str

class PaleoBondIntegration:
    """Manage integration with PaleoBond-PCP project"""
    
    def check_api_compatibility(self, current_version: str) -> List[PaleoBondCompatibilityIssue]:
        """Check for API compatibility issues"""
        # This would implement actual compatibility checking
        return []
    
    def create_sync_issue(self, issue_type: str, details: Dict) -> str:
        """Create synchronization issue between projects"""
        issue_body = f"""**PaleoBond-PCP Integration Issue**

**Issue Type:** {issue_type}

**Details:**
{json.dumps(details, indent=2)}

**Coordination Required:**
- [ ] Notify PaleoBond-PCP maintainers
- [ ] Test integration compatibility  
- [ ] Update API documentation
- [ ] Coordinate release timing
- [ ] Verify deployment compatibility

**Impact Assessment:**
- polyID Version: {details.get('polyid_version', 'current')}
- PaleoBond-PCP Impact: {details.get('impact_level', 'TBD')}

*This issue requires coordination between polyID and PaleoBond-PCP teams.*
"""
        return issue_body

# Integration monitoring
def monitor_paleobond_integration():
    """Monitor integration health with PaleoBond-PCP"""
    integration = PaleoBondIntegration()
    
    # Check compatibility
    issues = integration.check_api_compatibility("current")
    
    if issues:
        for issue in issues:
            # Create project tracking item
            details = {
                'api_version': issue.api_version,
                'breaking_change': issue.breaking_change,
                'impact_level': issue.impact_level,
                'affected_components': issue.affected_components
            }
            issue_body = integration.create_sync_issue("API Compatibility", details)
            # This would create actual GitHub issue
```

## Summary

This comprehensive template collection provides:

✅ **Complete automation workflows** for issue routing, PR handling, and CI/CD integration
✅ **Setup scripts** for quick project creation and configuration  
✅ **Field configuration** templates for all three project boards
✅ **Integration examples** for performance monitoring, research workflow, and PaleoBond-PCP coordination
✅ **Maintenance procedures** and monitoring scripts
✅ **Best practices** for open source project management

All files can be implemented by copying the templates and adjusting the project numbers, repository details, and authentication tokens as needed.