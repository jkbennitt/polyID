"""
PolyID - Enhanced Gradio Interface with Advanced Features
Advanced user interface with molecular visualization, batch processing, and performance monitoring
"""

import gradio as gr
import pandas as pd
import numpy as np
import time
import io
from typing import Dict, List, Tuple, Any, Optional

# Import our enhanced components
from polyid.molecular_viewer import MolecularViewer
from polyid.async_processor import get_async_processor
from polyid.performance_monitor import get_performance_monitor
from polyid.optimized_predictor import OptimizedPredictor
from polyid.cache_manager import get_cache_manager

class EnhancedPolyIDInterface:
    """Enhanced Gradio interface with advanced features"""
    
    def __init__(self):
        self.molecular_viewer = MolecularViewer()
        self.async_processor = get_async_processor()
        self.performance_monitor = get_performance_monitor()
        self.predictor = OptimizedPredictor()
        self.cache_manager = get_cache_manager()
        
        # Sample polymers
        self.sample_polymers = {
            "Polyethylene (PE)": "CC",
            "Polypropylene (PP)": "CC(C)",
            "Polystyrene (PS)": "CC(c1ccccc1)",
            "Poly(methyl methacrylate) (PMMA)": "CC(C)(C(=O)OC)",
            "Polyethylene terephthalate (PET)": "COC(=O)c1ccc(C(=O)O)cc1.OCCO",
            "Polycarbonate (PC)": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1.O=C(Cl)Cl",
            "Nylon-6 (PA6)": "NCCCCCC(=O)",
            "Polyurethane": "CC(C)(C)OC(=O)NCCCCCCNC(=O)O"
        }
        
        self.available_properties = [
            "Glass Transition Temperature (Tg)",
            "Melting Temperature (Tm)",
            "Density",
            "Elastic Modulus"
        ]

    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface"""
        
        custom_css = """
        .performance-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-error { color: #f44336; }
        """
        
        with gr.Blocks(
            title="PolyID - Advanced Polymer Property Prediction",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1>🧬 PolyID - Advanced Polymer Analysis Platform</h1>
                <p style="font-size: 1.2em; color: #666;">
                    State-of-the-art polymer property prediction with interactive visualization and batch processing
                </p>
                <p style="color: #888;">Optimized Performance Pipeline | Real-time Analytics | Molecular Visualization</p>
            </div>
            """)
            
            with gr.Tabs() as main_tabs:
                
                # Tab 1: Single Prediction with Enhanced Visualization
                with gr.Tab("🔬 Single Prediction", elem_id="single-tab"):
                    self._create_single_prediction_tab()
                
                # Tab 2: Batch Processing
                with gr.Tab("📊 Batch Processing", elem_id="batch-tab"):
                    self._create_batch_processing_tab()
                
                # Tab 3: Performance Dashboard
                with gr.Tab("📈 Performance Dashboard", elem_id="performance-tab"):
                    self._create_performance_dashboard_tab()
                
                # Tab 4: Model Analysis
                with gr.Tab("🧠 Model Analysis", elem_id="analysis-tab"):
                    self._create_model_analysis_tab()
                
                # Tab 5: Help & Documentation
                with gr.Tab("📚 Documentation", elem_id="help-tab"):
                    self._create_help_tab()
            
            # Global status footer
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;">
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div><strong>🚀 Optimized Performance</strong><br><small>Advanced caching & async processing</small></div>
                        <div><strong>📊 Real-time Monitoring</strong><br><small>Performance analytics & optimization</small></div>
                        <div><strong>🎨 Interactive Visualization</strong><br><small>Molecular structure & property analysis</small></div>
                        <div><strong>⚡ Batch Processing</strong><br><small>High-throughput predictions</small></div>
                    </div>
                </div>
                """)
        
        return interface
    
    def _create_single_prediction_tab(self):
        """Create enhanced single prediction interface"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔬 Input Parameters")
                
                # SMILES input
                smiles_input = gr.Textbox(
                    label="Polymer SMILES String",
                    placeholder="Enter SMILES (e.g., CC for polyethylene)",
                    value="CC",
                    info="Enter the SMILES representation of your polymer"
                )
                
                # Sample polymer selector
                sample_dropdown = gr.Dropdown(
                    choices=list(self.sample_polymers.keys()),
                    label="Select Sample Polymer",
                    value=None,
                    info="Choose from common polymer examples"
                )
                
                # Property selection with enhanced options
                property_checkboxes = gr.CheckboxGroup(
                    choices=self.available_properties,
                    label="Properties to Predict",
                    value=["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"],
                    info="Select one or more properties to predict"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    use_cache_checkbox = gr.Checkbox(
                        label="Use Caching",
                        value=True,
                        info="Enable result caching for faster repeated predictions"
                    )
                    
                    visualization_type = gr.Radio(
                        choices=["Radar Chart", "Bar Chart", "Both"],
                        label="Visualization Type",
                        value="Both",
                        info="Choose how to display results"
                    )
                
                predict_button = gr.Button("🔬 Predict Properties", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results & Visualization")
                
                # Status and validation
                status_output = gr.HTML(label="Status")
                
                with gr.Row():
                    # Molecular structure visualization
                    with gr.Column():
                        molecular_image = gr.Image(
                            label="Molecular Structure",
                            type="filepath",
                            interactive=False
                        )
                        
                        descriptors_plot = gr.Plot(
                            label="Molecular Descriptors"
                        )
                    
                    with gr.Column():
                        # Property visualization
                        properties_radar = gr.Plot(
                            label="Property Radar Chart"
                        )
                        
                        properties_bar = gr.Plot(
                            label="Property Comparison"
                        )
                
                # Detailed results
                detailed_results = gr.JSON(
                    label="Detailed Prediction Results",
                    show_label=True
                )
                
                # Performance metrics
                prediction_metrics = gr.HTML(
                    label="Prediction Performance"
                )
        
        # Event handlers
        def update_smiles_from_sample(sample_name):
            if sample_name and sample_name in self.sample_polymers:
                return self.sample_polymers[sample_name]
            return gr.update()
        
        def predict_with_visualization(smiles, properties, use_cache, viz_type):
            return self._enhanced_predict_single(smiles, properties, use_cache, viz_type)
        
        sample_dropdown.change(
            fn=update_smiles_from_sample,
            inputs=[sample_dropdown],
            outputs=[smiles_input]
        )
        
        predict_button.click(
            fn=predict_with_visualization,
            inputs=[smiles_input, property_checkboxes, use_cache_checkbox, visualization_type],
            outputs=[status_output, molecular_image, descriptors_plot, properties_radar, 
                    properties_bar, detailed_results, prediction_metrics]
        )
    
    def _create_batch_processing_tab(self):
        """Create batch processing interface"""
        
        gr.Markdown("### 📊 Batch Processing - High-throughput Predictions")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Upload Data")
                
                file_upload = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    info="Upload a CSV file with SMILES data"
                )
                
                smiles_column = gr.Textbox(
                    label="SMILES Column Name",
                    value="smiles_polymer",
                    info="Name of column containing SMILES strings"
                )
                
                batch_properties = gr.CheckboxGroup(
                    choices=self.available_properties,
                    label="Properties to Predict",
                    value=["Glass Transition Temperature (Tg)", "Melting Temperature (Tm)"],
                    info="Select properties for batch prediction"
                )
                
                batch_process_button = gr.Button("🚀 Process Batch", variant="primary")
                
                with gr.Accordion("Export Options", open=False):
                    export_format = gr.Radio(
                        choices=["CSV", "Excel", "JSON"],
                        label="Export Format",
                        value="CSV"
                    )
                    
                    download_button = gr.DownloadButton(
                        label="📥 Download Results",
                        variant="secondary"
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("#### Processing Status & Results")
                
                batch_progress = gr.Progress()
                batch_status = gr.HTML()
                
                results_preview = gr.Dataframe(
                    label="Results Preview",
                    interactive=False,
                    max_rows=10
                )
                
                batch_statistics = gr.HTML(
                    label="Processing Statistics"
                )
        
        # Batch processing handler
        def process_batch_file(file_obj, smiles_col, properties, format_type):
            return self._process_batch_async(file_obj, smiles_col, properties, format_type, batch_progress)
        
        batch_process_button.click(
            fn=process_batch_file,
            inputs=[file_upload, smiles_column, batch_properties, export_format],
            outputs=[batch_status, results_preview, batch_statistics, download_button]
        )
    
    def _create_performance_dashboard_tab(self):
        """Create performance monitoring dashboard"""
        
        gr.Markdown("### 📈 Performance Dashboard - Real-time Analytics")
        
        with gr.Row():
            with gr.Column():
                current_metrics_display = gr.HTML(
                    label="Current Performance Metrics"
                )
                
                performance_chart = gr.Plot(
                    label="Performance Trends"
                )
            
            with gr.Column():
                recommendations_display = gr.HTML(
                    label="Optimization Recommendations"
                )
                
                system_status = gr.HTML(
                    label="System Status"
                )
        
        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh Metrics", variant="secondary")
            export_metrics_button = gr.Button("📊 Export Metrics", variant="outline")
            
        def refresh_dashboard():
            return self._get_dashboard_data()
        
        def export_performance_data():
            return self._export_performance_metrics()
        
        # Auto-refresh every 30 seconds
        refresh_button.click(
            fn=refresh_dashboard,
            outputs=[current_metrics_display, performance_chart, recommendations_display, system_status]
        )
        
        export_metrics_button.click(
            fn=export_performance_data,
            outputs=[gr.File()]
        )
    
    def _create_model_analysis_tab(self):
        """Create model analysis and comparison interface"""
        
        gr.Markdown("### 🧠 Model Analysis - Deep Insights")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Cache Analysis")
                cache_stats = gr.HTML()
                cache_chart = gr.Plot(label="Cache Performance")
                
                clear_cache_button = gr.Button("🗑️ Clear Cache", variant="secondary")
                
            with gr.Column():
                gr.Markdown("#### Prediction Analysis")
                prediction_stats = gr.HTML()
                accuracy_chart = gr.Plot(label="Prediction Distribution")
        
        def analyze_cache_performance():
            stats = self.cache_manager.get_cache_stats()
            return self._format_cache_analysis(stats)
        
        def clear_system_cache():
            self.cache_manager.clear_cache()
            return "Cache cleared successfully"
        
        clear_cache_button.click(
            fn=clear_system_cache,
            outputs=[gr.HTML()]
        )
    
    def _create_help_tab(self):
        """Create help and documentation tab"""
        
        gr.Markdown("""
        # 📚 PolyID Documentation & Help
        
        ## 🚀 Quick Start Guide
        
        ### Single Predictions
        1. Enter a polymer SMILES string or select from sample polymers
        2. Choose properties to predict
        3. Click "Predict Properties" to get results with interactive visualizations
        
        ### Batch Processing
        1. Upload a CSV file with SMILES data
        2. Specify the column containing SMILES strings
        3. Select properties to predict
        4. Process and download results in your preferred format
        
        ### Performance Dashboard
        - Monitor real-time system performance
        - Get optimization recommendations
        - Track prediction accuracy and throughput
        
        ## 🔬 Supported Properties
        
        - **Glass Transition Temperature (Tg)**: Temperature range where polymer transitions from glassy to rubbery state
        - **Melting Temperature (Tm)**: Temperature at which crystalline regions melt
        - **Density**: Mass per unit volume of the polymer
        - **Elastic Modulus**: Measure of material stiffness
        
        ## 📊 Features
        
        - **Advanced Caching**: Intelligent result caching for faster predictions
        - **Async Processing**: Concurrent batch processing for high throughput
        - **Interactive Visualization**: Molecular structure and property visualizations
        - **Performance Monitoring**: Real-time performance analytics
        - **Export Options**: Multiple formats (CSV, Excel, JSON)
        
        ## 🏗️ System Architecture
        
        - **Optimized Predictor**: High-performance prediction pipeline
        - **Molecular Viewer**: Interactive structure visualization
        - **Performance Monitor**: Real-time system analytics
        - **Async Processor**: Concurrent batch processing
        
        ## 📞 Support
        
        For technical support or questions:
        - Check performance dashboard for system status
        - Review optimization recommendations
        - Ensure input SMILES are valid
        
        ---
        **PolyID v2.0** - Advanced Polymer Property Prediction Platform
        """)
    
    def _enhanced_predict_single(self, smiles: str, properties: List[str], 
                               use_cache: bool, viz_type: str) -> Tuple:
        """Enhanced single prediction with comprehensive visualization"""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not smiles or not smiles.strip():
                return self._create_error_response("Please enter a SMILES string")
            
            if not properties:
                return self._create_error_response("Please select at least one property")
            
            # Perform prediction
            result = self.predictor.predict_properties(smiles.strip(), properties, use_cache)
            
            if 'error' in result:
                return self._create_error_response(f"Prediction failed: {result['error']}")
            
            # Record performance metrics
            prediction_time = time.time() - start_time
            cache_hit = 'cached' in str(result).lower()
            
            self.performance_monitor.record_prediction_performance(
                prediction_time, cache_hit, False, smiles
            )
            
            # Generate visualizations
            molecular_img = self.molecular_viewer.create_molecule_image(smiles)
            descriptors_plot = self.molecular_viewer.create_molecular_descriptors_chart(smiles)
            
            # Property visualizations based on user choice
            radar_plot = None
            bar_plot = None
            
            if viz_type in ["Radar Chart", "Both"]:
                radar_plot = self.molecular_viewer.create_property_radar_chart(result)
            
            if viz_type in ["Bar Chart", "Both"]:
                bar_plot = self.molecular_viewer.create_property_comparison_chart(result)
            
            # Format status
            status_html = f"""
            <div class="performance-card">
                <h3>✅ Prediction Successful</h3>
                <p><strong>SMILES:</strong> {smiles}</p>
                <p><strong>Properties:</strong> {len(properties)} predicted</p>
                <p><strong>Processing Time:</strong> {prediction_time:.3f}s</p>
                <p><strong>Cache Hit:</strong> {'Yes' if cache_hit else 'No'}</p>
            </div>
            """
            
            # Performance metrics
            metrics_html = f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <h4>📊 Performance Metrics</h4>
                <div style="display: flex; justify-content: space-between;">
                    <div><strong>Processing Time:</strong> {prediction_time:.3f}s</div>
                    <div><strong>Cache Status:</strong> {'Hit' if cache_hit else 'Miss'}</div>
                    <div><strong>Properties:</strong> {len(result)}</div>
                </div>
            </div>
            """
            
            return (
                status_html,
                molecular_img,
                descriptors_plot,
                radar_plot,
                bar_plot,
                result,
                metrics_html
            )
            
        except Exception as e:
            self.performance_monitor.record_prediction_performance(
                time.time() - start_time, False, True, smiles
            )
            return self._create_error_response(f"Unexpected error: {str(e)}")
    
    def _create_error_response(self, message: str) -> Tuple:
        """Create standardized error response"""
        
        error_html = f"""
        <div style="background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px; border-left: 4px solid #c62828;">
            <h3>❌ Error</h3>
            <p>{message}</p>
        </div>
        """
        
        return (error_html, None, None, None, None, {}, "")
    
    def _process_batch_async(self, file_obj, smiles_col: str, properties: List[str], 
                           format_type: str, progress) -> Tuple:
        """Process batch file asynchronously"""
        
        if not file_obj:
            return "Please upload a CSV file", None, "", None
        
        try:
            # Read file content
            if hasattr(file_obj, 'name'):
                with open(file_obj.name, 'r') as f:
                    file_content = f.read()
            else:
                file_content = str(file_obj)
            
            # Process with progress tracking
            def progress_callback(progress_val, processed, total):
                progress(progress_val, f"Processed {processed}/{total} predictions")
            
            # Process batch
            results_df = self.async_processor.process_csv_file(
                file_content, smiles_col, properties, progress_callback
            )
            
            # Generate statistics
            total_predictions = len(results_df)
            successful_predictions = len(results_df.dropna(subset=[f"{prop}_pred" for prop in properties]))
            success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0
            
            stats_html = f"""
            <div class="performance-card">
                <h3>📊 Batch Processing Complete</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div><strong>Total Predictions:</strong> {total_predictions}</div>
                    <div><strong>Successful:</strong> {successful_predictions}</div>
                    <div><strong>Success Rate:</strong> {success_rate:.1%}</div>
                    <div><strong>Format:</strong> {format_type}</div>
                </div>
            </div>
            """
            
            status_html = """
            <div style="background: #e8f5e8; color: #2e7d32; padding: 15px; border-radius: 8px;">
                <h3>✅ Batch Processing Successful</h3>
                <p>Your batch predictions have been completed. Review the results below and download when ready.</p>
            </div>
            """
            
            # Prepare download
            export_data = self.async_processor.export_results(results_df, format_type.lower())
            filename = f"polyid_batch_results_{int(time.time())}.{format_type.lower()}"
            
            return status_html, results_df.head(10), stats_html, gr.DownloadButton(
                label="📥 Download Results",
                value=export_data,
                filename=filename
            )
            
        except Exception as e:
            error_html = f"""
            <div style="background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px;">
                <h3>❌ Batch Processing Failed</h3>
                <p>{str(e)}</p>
            </div>
            """
            return error_html, None, "", None
    
    def _get_dashboard_data(self) -> Tuple:
        """Get performance dashboard data"""
        
        current_metrics = self.performance_monitor.get_current_metrics()
        recommendations = self.performance_monitor.get_optimization_recommendations()
        
        # Format current metrics
        metrics_html = f"""
        <div class="performance-card">
            <h3>📊 Current Performance</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div>
                    <div class="metric-value">{current_metrics['system']['cpu_usage_percent']:.1f}%</div>
                    <div>CPU Usage</div>
                </div>
                <div>
                    <div class="metric-value">{current_metrics['system']['memory_usage_percent']:.1f}%</div>
                    <div>Memory Usage</div>
                </div>
                <div>
                    <div class="metric-value">{current_metrics['application']['cache_hit_rate']:.1%}</div>
                    <div>Cache Hit Rate</div>
                </div>
                <div>
                    <div class="metric-value">{current_metrics['application']['prediction_count_last_minute']}</div>
                    <div>Predictions/Min</div>
                </div>
            </div>
        </div>
        """
        
        # Format recommendations
        rec_html = "<h3>💡 Optimization Recommendations</h3>"
        if recommendations:
            for rec in recommendations[:3]:  # Show top 3
                priority_class = f"status-{rec['priority'].lower()}"
                rec_html += f"""
                <div style="margin: 10px 0; padding: 10px; border-left: 4px solid; border-color: var(--{rec['priority'].lower()}-color);">
                    <strong class="{priority_class}">{rec['category']} - {rec['priority']} Priority</strong><br>
                    <strong>Issue:</strong> {rec['issue']}<br>
                    <strong>Recommendation:</strong> {rec['recommendation']}
                </div>
                """
        else:
            rec_html += "<p style='color: #4CAF50;'>✅ No optimization recommendations - system running optimally!</p>"
        
        return metrics_html, None, rec_html, "System Status: Operational"