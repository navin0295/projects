import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:logging/logging.dart';

class Prompt extends StatefulWidget {
  const Prompt({super.key});

  @override
  State<Prompt> createState() => _PromptState();
}

class _PromptState extends State<Prompt> {
  final promptController = TextEditingController();
  final logger = Logger('ChatLogger'); // Initialize Logger
  List<String> chatHistory = [];
  bool isInitialPosition = true;
  bool isTyping = false;

  @override
  void initState() {
    super.initState();
    Logger.root.level = Level.ALL; // Set logging level
    Logger.root.onRecord.listen((record) {
      print('${record.level.name}: ${record.time}: ${record.message}');
    });
  }

  Future<void> chat(String prompt) async {
    if (prompt.isEmpty) {
      return;
    }

    setState(() {
      if (promptController.text.isNotEmpty) {
        chatHistory.add("You: $prompt");
        promptController.clear();
        isTyping = true;
      }
    });

    try {
      print("User prompt: $prompt"); // Debugging print
      // Set `backendUrl` to your running backend (ngrok or other).
      // Do NOT commit a real ngrok URL to the repo; keep it local or
      // configure via environment when running the app.
      final String backendUrl = 'https://YOUR_BACKEND_URL_HERE';
      final response = await http.post(
        Uri.parse('$backendUrl/api'),
        headers: <String, String>{
          'accept': 'application/json',
          "Content-Type": "application/json; charset=UTF-8"
        },
        body: jsonEncode(<String, String>{
          "input": prompt,
        }),
      ).timeout(Duration(minutes: 6));

      print("Response: ${response.body}"); // Print full response for debugging

      if (response.statusCode == 200) {
        final decodedResponse = jsonDecode(response.body);
        final botResponse = decodedResponse['response'] ?? decodedResponse['message'] ?? "I'm here to listen.";
        setState(() {
          chatHistory.add("finbot: $botResponse");
        });
        logger.info("$botResponse");
      } else {
        setState(() {
          chatHistory.add("finbot: Something went wrong. Please try again.");
        });
        logger.warning("Failed to get response: ${response.statusCode}");
      }
    } on TimeoutException {
      setState(() {
        chatHistory.add("The request timed out after 2 minutes.");
      });
      logger.severe("Request timed out.");
    } on SocketException {
      print('Socket error');
    } catch (error) {
      setState(() {
        chatHistory.add("Error occurred. Check your connection.");
      });
      logger.severe("Error: $error");
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Spacer(),
            const Text(
              "FINbot",
              style: TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.w500,
              ),
            ),
            const Spacer(),
            IconButton(
              onPressed: () {
                Navigator.pushReplacementNamed(context, '/');
              },
              icon: const Icon(
                Icons.logout,
                color: Colors.white,
              ),
            ),
          ],
        ),
        backgroundColor: Colors.teal,
        automaticallyImplyLeading: false,
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(25.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                //const CircleAvatar(
                // radius: 50,

                //  ),
                const SizedBox(height: 28),
                SizedBox(
                  width: 400,
                  child: const Text(
                    'Hey I\'m finbot, your AI friend for financial stock advice.\nWhats on your mind?',
                    style: TextStyle(fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                ),
                const SizedBox(height: 20),
                // Chat history list
                Container(
                  height: MediaQuery.sizeOf(context).height -
                      450, // Limit height for scroll
                  child: ListView.builder(
                    itemCount: chatHistory.length,
                    itemBuilder: (context, index) {
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4.0),
                        child: Align(
                          alignment: chatHistory[index].startsWith("You:")
                              ? Alignment.centerRight
                              : Alignment.centerLeft,
                          child: Container(
                            padding: const EdgeInsets.all(12),
                            width: 500,
                            decoration: BoxDecoration(
                              color: chatHistory[index].startsWith("You:")
                                  ? Colors.teal[100]
                                  : Colors.grey[300],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              chatHistory[index],
                              softWrap: true,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 20),
                // Text input field and send button
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: TextField(
                    controller: promptController,
                    decoration: InputDecoration(
                      hintText: "Type your message",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () {
                    chat(promptController.text);
                  },
                  child: const Text('Send'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
