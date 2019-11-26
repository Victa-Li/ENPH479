/*
 * Reading digital outputs from Optira Series PI encoder
 * Date: 2019 Nov 25
 * Author: ENPH 479 
 * Board: Teensy 3.6
 */
#include <Encoder.h>

// 1. Hardware wiring:
// Optira encoder has A+, A- (A) and B+ B- (B) for digital A-quad-B reading.
//   Best Performance: both pins have interrupt capability
//   Good Performance: only the first pin has interrupt capability
//   Low Performance:  neither pin has interrupt capability

// Change these two numbers to the pins connected to your encoder.
#define CHANNEL_A_PLUS 0
#define CHANNEL_A_MINUS 1
#define CHANNEL_B_PLUS 2
#define CHANNEL_B_MINUS 3

void setup() {
  pinMode (CHANNEL_A_PLUS, INPUT);
  pinMode (CHANNEL_A_MINUS, INPUT);
  pinMode (CHANNEL_B_PLUS, INPUT);
  pinMode (CHANNEL_B_MINUS, INPUT);
  Serial.begin(9600);
  Serial.println("Optira Series PI Encoder Test:");
}

// 2. Define and initialize variables 
Encoder channelA(CHANNEL_A_PLUS, CHANNEL_A_MINUS);
Encoder channelB(CHANNEL_B_PLUS, CHANNEL_B_MINUS);
long positionA  = 0;
long positionB = 0;
int encoder0Pos = 0;

// 3. Read from encoder
void loop() {
  long newA, newB;
  newA = channelA.read();
  newB = channelB.read();

  // 3.a Verify readings (debugging only)
  int aPlue = digitalRead(CHANNEL_A_PLUS);
  int aMinus = digitalRead(CHANNEL_A_MINUS);
  int bPlue = digitalRead(CHANNEL_B_PLUS);
  int bMinus = digitalRead(CHANNEL_B_MINUS);  
  Serial.print("Channel A + : ");
  Serial.println(aPlue);
  Serial.print("Channel A - : ");
  Serial.println(aMinus);
  Serial.print("Channel B + : ");
  Serial.println(bPlue);
  Serial.print("Channel B - : ");
  Serial.println(bMinus);
  
  // 3.b Read without interrupt
  if (newA != positionA || newB != positionB) {
    Serial.print("Channel A = ");
    Serial.print(newA);
    Serial.print(", Channel B = ");
    Serial.print(newB);
    Serial.println();
    positionA = newA;
    positionB = newB;
  }

  // 3.c Debug with reset
  // if a character is sent from the serial monitor,
  // reset both back to zero.
  if (Serial.available()) {
    Serial.read();
    Serial.println("Reset both channels to zero");
    channelA.write(0);
    channelB.write(0);
  }
}
