# GitHub Projects Quick Reference Guide

> 🚀 **Daily cheat sheet for polyID GitHub Projects** - Keep this handy for efficient project management!

## 🔗 Quick Links

| Resource | Link | Purpose |
|----------|------|---------|
| 🔬 Research Board | `https://github.com/users/USERNAME/projects/1` | Scientific research & model development |
| 💻 Development Board | `https://github.com/users/USERNAME/projects/2` | Features, bugs, infrastructure |
| 🚀 Release Board | `https://github.com/users/USERNAME/projects/3` | Version planning & releases |
| 📊 Dashboard | [PROJECT_DASHBOARD.md](.github/PROJECT_DASHBOARD.md) | Project health & metrics |

## ⚡ Daily Commands (5-minute routine)

### Morning Setup
```bash
# Check your assigned items
gh issue list --assignee @me --state open

# View development board ready items  
gh project item-list 2 --format json | jq '.[] | select(.status == "Ready")'

# Check CI status
gh workflow list --all
```

### Quick Status Updates
```bash
# Move item to In Progress
gh project item-edit 2 ITEM_ID --field-id STATUS_FIELD_ID --text "In Progress"

# Add yourself to an item
gh issue edit ISSUE_NUMBER --add-assignee @me

# Create quick bug report
gh issue create --template bug_report.md --label bug
```

## 🏃‍♂️ Common Workflows

### 1. Research Task (2 minutes to create)
```markdown
Title: Literature Review - [Topic]
Labels: research:literature, priority:medium
Board: Research
Fields: Research Phase → Literature Review, Complexity → Simple (1-3 days)
```

### 2. Bug Report (1 minute to create)
```markdown
Title: [BUG] Brief description
Labels: bug, priority:high
Board: Development  
Fields: Development Type → Bug Fix, Component → [relevant area]
```

### 3. Performance Issue (automated)
- Performance monitoring creates these automatically
- Check Development Board for performance alerts
- Priority set based on impact severity

### 4. Release Planning (weekly)
```markdown
Title: Release v1.X.X Planning
Labels: release, planning
Board: Release Management
Fields: Release Version → v1.X.X, Release Type → [Patch/Minor/Major]
```

## 📋 Field Quick Reference

### Research Board Fields
- **Research Phase**: Literature Review → Hypothesis/Design → Development → Validation → Validated → Documentation
- **Scientific Impact**: High (Novel) | Medium (Improvement) | Low (Maintenance)
- **Complexity**: Simple (1-3 days) | Moderate (1-2 weeks) | Complex (2-4 weeks) | Research Project (1-3 months)

### Development Board Fields  
- **Development Type**: Feature | Bug Fix | Infrastructure | Documentation | Performance | Security
- **Component**: Core ML | Interface | Performance | Testing | Documentation | Infrastructure | Deployment
- **Technical Complexity**: Low (<1 day) | Medium (1-3 days) | High (3-7 days) | Epic (>1 week)

### Release Board Fields
- **Release Type**: Patch (Bug fixes) | Minor (New features) | Major (Breaking changes)
- **Deployment Target**: PyPI | TestPyPI | GitHub Release | HF Spaces | Docker Hub

## 🎯 Status Workflow

### Research Workflow
```
📚 Literature Review → 🔬 Hypothesis/Design → 🧪 Development → 📊 Validation → ✅ Validated → 📝 Documentation
```

### Development Workflow  
```
📋 Backlog → 🎯 Ready → 🔨 In Progress → 🔍 Code Review → 🧪 Testing → ✅ Done
```

### Release Workflow
```
📋 Planning → 🔨 Development → 🧪 Testing → 🚀 Ready to Release → ✅ Released → 📊 Post-Release
```

## 🏷️ Label System

### Priority Labels
- `priority:critical` - System down, security issue
- `priority:high` - User-impacting, performance issue
- `priority:medium` - Important feature, non-blocking bug
- `priority:low` - Nice-to-have, minor improvement

### Research Labels
- `research:literature` - Literature review tasks
- `research:modeling` - ML model development  
- `research:validation` - Scientific validation
- `research:analysis` - Data analysis
- `research:publication` - Documentation/papers

### Development Labels
- `good first issue` - Beginner-friendly tasks
- `help wanted` - Community contribution welcome
- `performance` - Performance-related issues
- `ci-failure` - CI/CD problems (auto-generated)

## ⚡ GitHub CLI Shortcuts

### Project Management
```bash
# Quick project overview
alias polyid-projects='gh project list --owner USERNAME'
alias polyid-research='gh project view 1'
alias polyid-dev='gh project view 2'
alias polyid-release='gh project view 3'

# Create common issue types
alias bug-report='gh issue create --template bug_report.md'
alias feature-request='gh issue create --template feature_request.md'

# Check your work
alias my-issues='gh issue list --assignee @me'
alias my-prs='gh pr list --author @me'
```

### Automation Testing
```bash
# Test project automation
gh issue create --title "Test: Automation Check" --label "test,automation" --body "Testing project routing"

# Check webhook deliveries
gh api repos/OWNER/REPO/hooks --jq '.[].deliveries_url'
```

## 🔄 Daily Routine Checklist

### ☀️ Morning (5 minutes)
- [ ] Check [Project Dashboard](.github/PROJECT_DASHBOARD.md)
- [ ] Review overnight CI failures or alerts
- [ ] Update status of items you're working on
- [ ] Check assigned items across all boards
- [ ] Identify any blockers or dependencies

### 🏃‍♂️ During Work (ongoing)
- [ ] Update item status when you start/finish work
- [ ] Link related issues and PRs
- [ ] Add comments for progress updates
- [ ] Move items through workflow columns

### 🌅 End of Day (2 minutes)
- [ ] Update status of in-progress items
- [ ] Close completed items  
- [ ] Plan tomorrow's priorities
- [ ] Check for new assignments or mentions

## 🚨 Emergency Procedures

### CI/CD Failure
1. **Immediate**: Check the failure in project boards (auto-created issue)
2. **Investigate**: Review logs and identify root cause
3. **Fix**: Create hotfix branch if critical
4. **Update**: Move related project items to appropriate status

### Performance Alert  
1. **Check**: Development board for auto-created performance issue
2. **Assess**: Review performance dashboard for impact
3. **Prioritize**: Set appropriate priority based on user impact
4. **Monitor**: Track resolution and verify improvement

### Research Blocker
1. **Document**: Create issue describing the blocker
2. **Consult**: Reach out to team/community for input
3. **Alternative**: Consider alternative approaches
4. **Timeline**: Adjust research milestones if needed

## 🎨 Keyboard Shortcuts (GitHub Web)

### Project Boards
- `c` - Create new item
- `e` - Edit item
- `Enter` - Open item details
- `Tab` - Move between fields
- `Esc` - Close modals

### Issues
- `g` + `i` - Go to issues
- `c` - Create new issue  
- `l` - Add labels
- `a` - Add assignees
- `m` - Add milestones

## 📊 Quick Metrics Check

### Performance Health
```bash
# Check system performance
python -c "from polyid.performance_monitor import get_performance_monitor; print(get_performance_monitor().get_current_metrics())"

# Cache status
python -c "from polyid.cache_manager import get_cache_manager; print(f'Cache size: {get_cache_manager().cache_size}')"
```

### Project Progress
```bash
# Research progress this week
gh project item-list 1 --format json | jq '[.[] | select(.updated_at > "2024-01-01")] | length'

# Development velocity
gh project item-list 2 --format json | jq '[.[] | select(.status == "Done" and .updated_at > "2024-01-01")] | length'
```

## 🤝 Team Coordination

### Research Collaboration
- **Literature Reviews**: Tag team members for input
- **Model Development**: Create linked development issues
- **Validation**: Cross-reference with other researchers

### Development Coordination
- **Code Reviews**: Use PR integration for automatic updates
- **Bug Fixes**: Link to original feature development
- **Performance**: Coordinate with research validation

### PaleoBond-PCP Sync
- **Weekly**: Check integration board for coordination items
- **Releases**: Coordinate timing with main project
- **Issues**: Create cross-project coordination issues

## 🔧 Troubleshooting Quick Fixes

### Project Item Not Moving
```bash
# Check automation status
gh project rule-list PROJECT_NUMBER

# Manually move item
gh project item-edit PROJECT_NUMBER ITEM_ID --field-id STATUS_FIELD_ID --text "New Status"
```

### Missing Automation
```bash
# Re-run project automation workflow
gh workflow run project_automation.yml

# Check recent workflow runs
gh run list --workflow=project_automation.yml
```

### Performance Alerts Not Working
```python
# Test performance monitoring
from polyid.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
print("Current metrics:", monitor.get_current_metrics())
print("Alert thresholds:", monitor.get_alert_thresholds())
```

## 📱 Mobile Quick Actions

### GitHub Mobile App
- Check notifications for project updates
- Quick issue triage and labeling
- Status updates and comments
- PR reviews and approvals

### Browser Bookmarks (Mobile)
- Research Board (mobile view)
- Development Board (mobile view)  
- Quick issue creation
- Project dashboard

---

## 🎯 Success Metrics

**Daily Goals:**
- ✅ All assigned items have current status
- ✅ No items stuck >3 days without update
- ✅ Performance alerts addressed within 24h
- ✅ Research progress documented weekly

**Weekly Goals:**  
- ✅ Research board flowing smoothly through phases
- ✅ Development board maintaining velocity
- ✅ Release board aligned with roadmap
- ✅ Community contributions acknowledged

**Monthly Goals:**
- ✅ Project health dashboard green
- ✅ Automation rules optimized
- ✅ Performance benchmarks met
- ✅ PaleoBond-PCP coordination smooth

---

*Keep this guide handy - print it out or bookmark for daily reference!*

**Last Updated**: [Current Date]  
**Next Review**: [Monthly]

> 💡 **Tip**: Customize the USERNAME and project numbers in links, then bookmark this page for instant access to your polyID project management workflow!