import 'dart:convert';
import 'package:http/http.dart' as http;

class StockBotService {
  final String apiUrl = "http://localhost:3000/api";  // Change if hosted
  // Replace with your API key in a local-only config. Do NOT commit keys.
  final String apiKey = "YOUR_STOCK_API_KEY_HERE";

  Future<String> getStockData(String query) async {
    final response = await http.post(
      Uri.parse("$apiUrl/chat"),
      headers: {
        "Authorization": "Bearer $apiKey",
        "Content-Type": "application/json",
      },
      body: jsonEncode({"query": query}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body)["response"];
    } else {
      return "Error: ${response.statusCode} - ${response.body}";
    }
  }
}
