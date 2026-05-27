import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:fl_chart/fl_chart.dart';

class HomePageContent extends StatefulWidget {
  @override
  _HomePageContentState createState() => _HomePageContentState();
}

class _HomePageContentState extends State<HomePageContent> {
  // Set your Finnhub API key via a secure method before running.
  // DO NOT commit real API keys to source control. Replace the placeholder
  // below with your key in a local-only config or environment.
  final String apiKey = 'YOUR_FINNHUB_API_KEY_HERE';
  String? stockData;
  List<String> stockSymbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'INTC']; // Initial stock symbols
  List<FlSpot> stockPrices = [];
  List<FlSpot> purchasePrices = []; // List for purchase prices
  Map<String, double> purchasePriceMap = {}; // Map to store purchase price for each stock
  final TextEditingController symbolController = TextEditingController();
  final TextEditingController priceController = TextEditingController(); // Controller for purchase price

  @override
  void initState() {
    super.initState();
    fetchStockData(); // Fetch stock data when the widget is initialized
  }

  Future<void> fetchStockData() async {
    String fetchedData = '';
    List<FlSpot> prices = [];
    List<FlSpot> userPrices = []; // List for user purchase prices on the graph

    for (int i = 0; i < stockSymbols.length; i++) {
      String symbol = stockSymbols[i];
      final response = await http.get(Uri.parse(
          'https://finnhub.io/api/v1/quote?symbol=$symbol&token=$apiKey'));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // Fetch the current price (c) and add to prices
        double price = double.parse(data['c'].toString());
        prices.add(FlSpot(i.toDouble(), price));

        // If user has provided a purchase price, add to userPrices
        if (purchasePriceMap.containsKey(symbol)) {
          double purchasePrice = purchasePriceMap[symbol]!;
          userPrices.add(FlSpot(i.toDouble(), purchasePrice));
        } else {
          userPrices.add(FlSpot(i.toDouble(), price)); // Default to current price if no purchase price
        }

        fetchedData += 'The current price of $symbol is \$$price\n';
      } else {
        fetchedData += 'Error fetching data for $symbol.\n';
      }
    }

    setState(() {
      stockData = fetchedData; // Update the state with fetched stock data
      stockPrices = prices; // Update chart data
      purchasePrices = userPrices; // Update user purchase prices for chart
    });
  }

  void addStockSymbol() {
    final symbol = symbolController.text.trim().toUpperCase();
    final priceText = priceController.text.trim();
    double? purchasePrice = double.tryParse(priceText);

    if (symbol.isNotEmpty && !stockSymbols.contains(symbol) && purchasePrice != null) {
      setState(() {
        stockSymbols.add(symbol);
        purchasePriceMap[symbol] = purchasePrice; // Store purchase price
        fetchStockData(); // Refresh data after adding a new stock
      });
    }

    symbolController.clear(); // Clear the input field
    priceController.clear(); // Clear the purchase price field
  }

  void deleteStockSymbol(String symbol) {
    setState(() {
      stockSymbols.remove(symbol);
      purchasePriceMap.remove(symbol); // Remove the purchase price
      fetchStockData(); // Refresh data after removing a stock
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: symbolController,
                decoration: InputDecoration(
                  labelText: 'Enter stock symbol',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            Expanded(
              child: TextField(
                controller: priceController,
                decoration: InputDecoration(
                  labelText: 'Enter purchase price',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
              ),
            ),
            IconButton(
              icon: Icon(Icons.add),
              onPressed: addStockSymbol,
            ),
          ],
        ),
        Expanded(
          child: stockPrices.isNotEmpty
              ? LineChart(
            LineChartData(
              gridData: FlGridData(show: true),
              titlesData: FlTitlesData(
                leftTitles: SideTitles(
                    showTitles: true,
                    interval: 50,
                    getTitles: (value) {
                      return '\$${value.toInt()}';
                    },
                  reservedSize: 40,
                ),
                bottomTitles: SideTitles(showTitles: true),
              ),
              borderData: FlBorderData(
                show: true,
                border: Border.all(color: const Color(0xff37434d), width: 1),
              ),
              minX: 0,
              maxX: stockPrices.length.toDouble() - 1,
              minY: 0,
              maxY: stockPrices.map((e) => e.y).reduce((a, b) => a > b ? a : b) + 10,
              lineBarsData: [
                LineChartBarData(
                  spots: stockPrices,
                  isCurved: true,
                  colors: [Colors.blue],
                  belowBarData: BarAreaData(show: false),
                ),
                LineChartBarData(
                  spots: purchasePrices,
                  isCurved: true,
                  colors: [Colors.red], // Use red to differentiate
                  belowBarData: BarAreaData(show: false),
                  dashArray: [5, 5], // Dashed line for purchase prices
                ),
              ],
            ),
          )
              : Center(child: CircularProgressIndicator()), // Show loading indicator while fetching data
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            stockData != null ? stockData! : 'Loading stock data...',
            textAlign: TextAlign.center,
          ),
        ),
        Wrap(
          children: stockSymbols.map((symbol) => Chip(
            label: Text(symbol),
            deleteIcon: Icon(Icons.close),
            onDeleted: () => deleteStockSymbol(symbol),
          )).toList(),
        ),
      ],
    );
  }
}
