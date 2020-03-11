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
#define CHANNEL_A_PLUS 17
#define CHANNEL_A_MINUS 16
#define CHANNEL_B_PLUS 15
#define CHANNEL_B_MINUS 14
#define PULSE_PER_MM 20000 // 20um scale

// 2. Define and initialize variables 
//Encoder channelA(CHANNEL_A_PLUS, CHANNEL_A_MINUS);
//Encoder channelB(CHANNEL_B_PLUS, CHANNEL_B_MINUS);
int count = 0;
long pos = 0;

void setup() {
  pinMode (CHANNEL_A_PLUS, INPUT);
  pinMode (CHANNEL_A_MINUS, INPUT);
  pinMode (CHANNEL_B_PLUS, INPUT);
  pinMode (CHANNEL_B_MINUS, INPUT);
  attachInterrupt(digitalPinToInterrupt(CHANNEL_A_MINUS), updateCount, RISING);
  Serial.begin(9600);
  Serial.println("Optira Series PI Encoder Test:");
}

void updateCount() {
  count += 1;
}

// 3. Read from encoder
void loop() {
  int aPlus; 
  int aMin; 
  int bPlus;
  int bMin;
  aPlus = digitalRead(CHANNEL_A_PLUS);
  aMin = digitalRead(CHANNEL_A_MINUS);
  bPlus = digitalRead(CHANNEL_B_PLUS);
  bMin = digitalRead(CHANNEL_B_MINUS);
  Serial.print(aPlus);
  Serial.print(" ");
  Serial.print(aMin);
  Serial.print(" ");
  Serial.print(bPlus);
  Serial.print(" ");
  Serial.print(bMin);
  Serial.print(" ");
  Serial.println();
  delay(500);
  if (count >= 0.1*PULSE_PER_MM) {
    pos += count / (2 * PULSE_PER_MM);
    count = 0;
//      Serial.println("Position = ");
//      Serial.println(pos);
  } else {
//      Serial.println("Count = ");
//      Serial.println(count);
  }
}
