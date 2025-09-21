# PolyID Documentation

Welcome to the PolyID documentation! This guide provides comprehensive information about the polymer property prediction system, deployment options, analysis reports, and development guides.

## 📚 Documentation Structure

### 🚀 [Deployment](./deployment/)
Documentation for deploying PolyID on various platforms:

- **[Standard GPU Implementation Guide](./deployment/Option1_Standard_GPU_Implementation.md)** - Complete guide for deploying PolyID on Hugging Face Standard GPU Spaces with full chemistry stack support

### 🔍 [Analysis](./analysis/)
Technical analysis and compatibility studies:

- **[ZeroGPU Compatibility Analysis](./analysis/ZeroGPU_Compatibility_Analysis.md)** - Comprehensive analysis of ZeroGPU limitations and chemistry package compatibility issues

### 📖 [Guides](./guides/)
Development and workflow guides:

- **[Branch Management Commands](./guides/Branch_Management_Commands.md)** - Git workflow and branch management for PolyID development

### 📋 [Reviews](./reviews/)
Detailed technical reviews and assessments:

- **[Architecture Review](./reviews/ARCHITECTURE_REVIEW.md)** - System architecture analysis and recommendations
- **[Containerized Deployment Analysis](./reviews/CONTAINERIZED_DEPLOYMENT_ANALYSIS.md)** - Container deployment strategies and optimization
- **[Neural Architecture Review](./reviews/NEURAL_ARCHITECTURE_REVIEW.md)** - Neural network architecture evaluation and improvements
- **[Scientific Validation Review](./reviews/SCIENTIFIC_VALIDATION_REVIEW.md)** - Scientific methodology and validation approaches

## 🎯 Quick Start

### For Developers
1. **Setup**: Follow the development commands in the main [CLAUDE.md](../CLAUDE.md)
2. **Deployment**: Choose your deployment strategy from the [deployment guide](./deployment/Option1_Standard_GPU_Implementation.md)
3. **Testing**: Use the testing suite in [../hf-spaces-testing/](../hf-spaces-testing/)

### For Deployment
1. **Standard GPU Spaces**: Follow the [Standard GPU Implementation Guide](./deployment/Option1_Standard_GPU_Implementation.md)
2. **Review Requirements**: Check [ZeroGPU Analysis](./analysis/ZeroGPU_Compatibility_Analysis.md) for platform limitations

## 🔗 Related Resources

- **[Main Project Documentation](../CLAUDE.md)** - Core development guidance
- **[HF Spaces Testing Suite](../hf-spaces-testing/)** - Comprehensive testing tools for deployment validation
- **[Analysis Reports](../reports/)** - Generated analysis reports and performance metrics
- **[Original Tests](../tests/)** - Core PolyID unit tests

## 📊 Project Status

**Current Branch**: `standard-gpu-deployment`
**Target Platform**: Hugging Face Standard GPU Spaces
**Python Version**: 3.11
**Chemistry Stack**: RDKit, NFP, m2p with full compatibility

## 🤝 Contributing

When contributing to documentation:

1. **Deployment docs** → Add to `deployment/`
2. **Technical analysis** → Add to `analysis/`
3. **Development guides** → Add to `guides/`
4. **System reviews** → Add to `reviews/`

Always update this index when adding new documentation files.

---

*For technical support and development guidance, refer to the main [CLAUDE.md](../CLAUDE.md) file in the project root.*