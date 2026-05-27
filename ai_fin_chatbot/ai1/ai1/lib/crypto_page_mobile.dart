// lib/crypto_page_mobile.dart
import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class MobileCryptoPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Poe Bot'),
      ),
      body: WebView(
        initialUrl: 'https://poe.com/BotECDOQ95N8N',
        javascriptMode: JavascriptMode.unrestricted,
      ),
    );
  }
}
