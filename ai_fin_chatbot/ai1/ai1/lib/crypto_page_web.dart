// lib/crypto_page_web.dart
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'dart:html' as html;

class WebCryptoPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      // Register the view factory for the iframe
      // ignore: undefined_prefixed_name
      ui.platformViewRegistry.registerViewFactory(
        'webview',
            (int viewId) => html.IFrameElement()
          ..src = 'http://example.com/'
          ..style.border = 'none',
      );

      return Scaffold(
        appBar: AppBar(
          title: Text('Poe Bot'),
        ),
        body: HtmlElementView(viewType: 'webview'),
      );
    } else {
      return Center(child: Text('This feature is only available on the web.'));
    }
  }
}
