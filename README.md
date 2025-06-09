# Heat Storage Model

This project implements a heat storage model that calculates heat loss for an underground cavern heat storage system using actual operational data. The model processes temperature readings and power input/output values to provide insights into the efficiency and performance of the heat storage system.

## Project Structure

```
heat-storage-model
├── data
│   └── operational_data.csv      # Contains actual operational data for the heat storage system
├── src
│   ├── heatstorage_model.py       # Main logic for the heat storage model
│   └── utils.py                   # Utility functions for data processing and analysis
├── tests
│   └── test_heatstorage_model.py   # Unit tests for the heat storage model
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd heat-storage-model
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the heat storage model, you can run the `heatstorage_model.py` script. Ensure that the `operational_data.csv` file is populated with the necessary data.

## Testing

To run the unit tests, execute the following command:
```
pytest tests/test_heatstorage_model.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.