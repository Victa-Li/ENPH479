// Source: http://www.bristolwatch.com/arduino/arduino2.htm

/* 
Demonstrates use of rotary encoder for motor
direction and distance.
*/

#define CHA 17
#define CHB 15

volatile int master_count = 0; // universal count
volatile int bck = 0; // universal count
volatile int fwd = 0; // universal count
volatile byte INTFLAG1 = 0; // interrupt status flag

void setup() { 
  pinMode(CHA, INPUT);
  pinMode(CHB, INPUT);
  
  Serial.begin(9600); 
  
  attachInterrupt(CHA, flag, RISING);  
  //  attachInterrupt(CHB, fwdflag, RISING); 
}

void loop() {

    if (INTFLAG1)   {
         Serial.print(master_count / 20000);
         Serial.print(" ");
         Serial.println();
         delay(300);
       INTFLAG1 = 0; // clear flag
    } // end if


} // end loop

void flag() {
  INTFLAG1 = 1;
  // add 1 to count for CW
  if (digitalRead(CHA) && !digitalRead(CHB)) {
    master_count++ ;
    fwd++;
  }
  // subtract 1 from count for CCW
  if (digitalRead(CHA) && digitalRead(CHB)) {
    master_count-- ;
    bck++;
  } 
}

//void fwdflag() {
//  INTFLAG1 = 1;
//  // add 1 to count for forward
//  if (!digitalRead(CHA) && digitalRead(CHB)) {
//    master_count++ ;
//    fwd++;
//  }
//}
//
//void bckflag() {
//  INTFLAG1 = 1;
//  // subtract 1 from count for backward
//  if (digitalRead(CHA) && !digitalRead(CHB)) {
//    master_count-- ;
//    bck++;
//  } 
//}
