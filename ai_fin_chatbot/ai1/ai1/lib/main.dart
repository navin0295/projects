// lib/main.dart
import 'package:ai1/prompt.dart';
import 'package:flutter/foundation.dart'
    show kIsWeb; // Import to check if running on web.
import 'package:flutter/material.dart';
import 'crypto_page_mobile.dart'; // Import for mobile crypto page
import 'crypto_page_web.dart'; // Import for web crypto page
// Add imports for other pages here if they are in separate files
import 'home_page_content.dart';
import 'financial_expense_page.dart';
import 'prompt.dart';
import 'portfolio_optimization_page.dart';
void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Financial Advisor',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;

  final List<Widget> _children = [
    HomePageContent(),
    FinancialExpensePage(),
    Prompt(),
    kIsWeb ? WebCryptoPage() : MobileCryptoPage(), // Conditional use
    StockBotScreen(),
  ];

  void onTabTapped(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Financial Advisor'),
      ),
      body: _children[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: onTabTapped,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.money), label: 'Expenses'),
          BottomNavigationBarItem(icon: Icon(Icons.chat), label: 'Chatbot'),
          BottomNavigationBarItem(icon: Icon(Icons.currency_bitcoin), label: 'Crypto'),
          BottomNavigationBarItem(icon: Icon(Icons.trending_up), label: 'market analyzer'),
        ],
      ),
    );
  }
}
