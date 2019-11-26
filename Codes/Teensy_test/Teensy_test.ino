/*
 Name:		FTIR_Teensy.ino
 Created:	7/13/2018 11:54:25 AM
 Author:	tang-
*/

#define AP A3
#define AN A2
#define BP A1
#define BN A0


// the setup function runs once when you press reset or power the board
void setup() {
  
  // initialize USB-to-PC:  
  Serial.begin(115200);
 
}

// the loop function runs over and over again until power down or reset
void loop() {

  Serial.println(analogRead(AP) + " , " analogRead(AN) + " , " + analogRead(BP) + " , " analogRead(BN));

}
