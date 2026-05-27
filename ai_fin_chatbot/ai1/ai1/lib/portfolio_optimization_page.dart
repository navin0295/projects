import 'package:flutter/material.dart';
import 'stockbot_service.dart';

class StockBotScreen extends StatefulWidget {
  @override
  _StockBotScreenState createState() => _StockBotScreenState();
}

class _StockBotScreenState extends State<StockBotScreen> {
  final StockBotService stockBotService = StockBotService();
  String stockData = "Loading...";

  @override
  void initState() {
    super.initState();
    fetchStockData();
  }

  void fetchStockData() async {
    String data = await stockBotService.getStockData("AAPL stock price");
    setState(() {
      stockData = data;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("StockBot Data")),
      body: Center(child: Text(stockData)),
    );
  }
}
