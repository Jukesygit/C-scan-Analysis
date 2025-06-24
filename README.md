# C-Scan Composite Analysis

A modern web application for analyzing and visualizing composite C-scan data, built with NiceGUI.

## Features

- **Modern Web Interface**: Beautiful, responsive UI with dark theme and glass-morphism effects
- **Composite Analysis**: Build composite C-scan visualizations from multiple scan files
- **Single Scan Processing**: Process and visualize individual C-scan files
- **Advanced Visualization**: Color-coded heatmaps with customizable thickness ranges
- **Cloud-Ready**: Fully deployable to cloud platforms like Render, Heroku, or Vercel
- **File Upload**: Drag-and-drop file uploads for CSV scan logs and C-scan data files

## Technologies Used

- **NiceGUI**: Modern Python web framework
- **NumPy & Pandas**: Data processing and analysis
- **Matplotlib**: Scientific plotting and visualization
- **SciPy**: Advanced scientific computing

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Deployment

### Cloud Deployment (Render, Heroku, etc.)

1. Push your code to GitHub
2. Connect your repository to your cloud provider
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `python app.py`

### Local Network Access

The app runs on `0.0.0.0:8080` by default, making it accessible from other devices on your local network.

## Usage

1. **Upload Files**: Upload your CSV scan log and C-scan data files
2. **Configure Settings**: Adjust visualization and processing parameters
3. **Build Composite**: Generate composite visualizations from multiple scans
4. **Process Single Scan**: Analyze individual scan files
5. **Download Results**: Export your visualizations as high-quality images

## File Formats

- **CSV Files**: Scan log files with columns: "File Name", "X Start (mm)", "Y Start (mm)"
- **C-Scan Data**: Text files (.txt, .asc) containing scan matrix data

## License

This project is for educational and research purposes.
