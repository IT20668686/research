import 'dart:ui';
import 'package:kathaappa/Screens/ScreenTest/Homescreen.dart';
import 'package:kathaappa/Screens/ScreenTest/RecordScreen.dart';
import 'package:flutter/material.dart';
import 'package:kathaappa/utils/configr.dart';

class ListWords extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    double fem = 1.0; // Set the value of fem according to your requirements
    double ffem = 1.0; // Set the value of ffem according to your requirements

    return SafeArea(
      child: Scaffold(
        body: Container(
          width: double.infinity,
          decoration: BoxDecoration(
            color: const Color(0xffffffff),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              ClipRect(
                child: BackdropFilter(
                  filter: ImageFilter.blur(
                    sigmaX: 13.5914087296 * fem,
                    sigmaY: 13.5914087296 * fem,
                  ),
                  child: Container(
                    margin: EdgeInsets.fromLTRB(
                        0 * fem, 0 * fem, 0 * fem, 16 * fem),
                    padding: EdgeInsets.fromLTRB(
                        36.33 * fem, 14 * fem, 14.67 * fem, 0 * fem),
                    width: double.infinity,
                    height: 54 * fem,
                    decoration: BoxDecoration(
                      color: const Color(0xffffffff),
                    ),
                    child: Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: const Color(0xffffffff),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            width: 50 * fem,
                            height: 30 * fem,
                            child: ElevatedButton(
                              onPressed: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => HomeScreen(),
                                  ),
                                );
                              },
                              child: Image.asset(
                                Configr.back_icon,
                                width: 50 * fem,
                                height: 30 * fem,
                              ),
                              style: ElevatedButton.styleFrom(
                                padding: EdgeInsets.zero,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20.0),
                                ),
                              ),
                            ),
                          ),
                          Row(
                            children: [
                              Container(
                                width: 45 * fem,
                                height: 32 * fem,
                                child: ElevatedButton(
                                  onPressed: () {
                                    // Profile button pressed action
                                  },
                                  child: Image.asset(
                                    Configr.home_icon,
                                    width: 50 * fem,
                                    height: 27 * fem,
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    padding: EdgeInsets.zero,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(20.0),
                                    ),
                                  ),
                                ),
                              ),
                              SizedBox(width: 16 * fem),
                              // Add necessary spacing
                              Container(
                                width: 43 * fem,
                                height: 32 * fem,
                                child: ElevatedButton(
                                  onPressed: () {
                                    // Home button pressed action
                                  },
                                  child: Image.asset(
                                    Configr.profile_icon,
                                    width: 50 * fem,
                                    height: 27 * fem,
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    padding: EdgeInsets.zero,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(20.0),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              Expanded(
                child: SingleChildScrollView(
                  child: Container(
                    padding: EdgeInsets.fromLTRB(
                        64 * fem, 13 * fem, 57 * fem, 11 * fem),
                    width: double.infinity,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment(0.407, -1),
                        end: Alignment(-0.407, 1),
                        colors: <Color>[
                          Color(0xff24d0a7),
                          Color(0xff0e510d),
                        ],
                        stops: <double>[0, 1],
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Center(
                          child: Text(
                            Configr.word_pg,
                            style: TextStyle(
                              fontSize: 18 * ffem,
                              // Adjust the font size as per your needs
                              fontWeight: FontWeight.bold,
                              // Add any other text style properties you desire
                            ),
                          ),
                        ),
                        SizedBox(height: 16 * fem), // Add necessary spacing
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              Configr.select_any,
                              style: TextStyle(
                                fontSize: 16 * ffem,
                                // Adjust the font size as per your needs
                                fontWeight: FontWeight.bold,
                                color: Colors.green[900],
                                // Add any other text style properties you desire
                              ),
                            ),
                            SizedBox(width: 8 * fem), // Add necessary spacing
                          ],
                        ),
                        SizedBox(height: 16 * fem), // Add necessary spacing
                        Center(
                          child: Column(
                            children: [
                              Container(
                                margin: EdgeInsets.fromLTRB(
                                    0 * fem, 0 * fem, 23 * fem, 23 * fem),
                                padding: EdgeInsets.all(9 * fem),
                                width: 240 * fem,
                                height: 240 * fem,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment(0.407, -1),
                                    end: Alignment(-0.407, 1),
                                    colors: <Color>[
                                      Color(0xff3f13ae),
                                      Color(0xffca1ac7),
                                    ],
                                    stops: <double>[0, 1],
                                  ),
                                ),
                                child: Center(
                                  child: SizedBox(
                                    width: 170 * fem,
                                    height: 170 * fem,
                                    child: ElevatedButton(
                                      onPressed: () {
                                        Navigator.push(
                                          context,
                                          MaterialPageRoute(
                                            builder: (context) =>
                                                RecordScreen(),
                                          ),
                                        );
                                      },
                                      child: Image.asset(
                                        Configr.dog_list,
                                        width: 236 * fem,
                                        height: 252 * fem,
                                      ),
                                      style: ElevatedButton.styleFrom(
                                        padding: EdgeInsets.zero,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              Container(
                                margin: EdgeInsets.fromLTRB(
                                    0 * fem, 0 * fem, 23 * fem, 23 * fem),
                                padding: EdgeInsets.all(9 * fem),
                                width: 240 * fem,
                                height: 240 * fem,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment(0.407, -1),
                                    end: Alignment(-0.407, 1),
                                    colors: <Color>[
                                      Color(0xff3f13ae),
                                      Color(0xffca1ac7),
                                    ],
                                    stops: <double>[0, 1],
                                  ),
                                ),
                                child: Center(
                                  child: SizedBox(
                                    width: 180 * fem,
                                    height: 200 * fem,
                                    child: Image.asset(
                                      Configr.goat_list,
                                      width: 236 * fem,
                                      height: 252 * fem,
                                    ),
                                  ),
                                ),
                              ),
                              Container(
                                margin: EdgeInsets.fromLTRB(
                                    0 * fem, 0 * fem, 23 * fem, 23 * fem),
                                padding: EdgeInsets.all(9 * fem),
                                width: 240 * fem,
                                height: 240 * fem,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment(0.407, -1),
                                    end: Alignment(-0.407, 1),
                                    colors: <Color>[
                                      Color(0xff3f13ae),
                                      Color(0xffca1ac7),
                                    ],
                                    stops: <double>[0, 1],
                                  ),
                                ),
                                child: Center(
                                  child: SizedBox(
                                    width: 180 * fem,
                                    height: 200 * fem,
                                    child: Image.asset(
                                      Configr.spider_list,
                                      width: 236 * fem,
                                      height: 252 * fem,
                                    ),
                                  ),
                                ),
                              ),
                              Container(
                                margin: EdgeInsets.fromLTRB(
                                    0 * fem, 0 * fem, 23 * fem, 23 * fem),
                                padding: EdgeInsets.all(9 * fem),
                                width: 240 * fem,
                                height: 240 * fem,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment(0.407, -1),
                                    end: Alignment(-0.407, 1),
                                    colors: <Color>[
                                      Color(0xff3f13ae),
                                      Color(0xffca1ac7),
                                    ],
                                    stops: <double>[0, 1],
                                  ),
                                ),
                                child: Center(
                                  child: SizedBox(
                                    width: 180 * fem,
                                    height: 200 * fem,
                                    child: Image.asset(
                                      Configr.dog_list,
                                      width: 236 * fem,
                                      height: 252 * fem,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
