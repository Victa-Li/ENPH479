/*
 Name:		FTIR_Teensy.ino
 Created:	7/13/2018 11:54:25 AM
 Author:	tang-
*/

#define DIGITAL_TRIG 11
#define LED_PIN 13
#define BUTTON_PIN 6
#define INT_PIN A19 //intensity pin
#define REFINT_PIN A0 //reference intensity pin (HeNe)

int counter, refIntV, intV;
char outgoingChar, incomingChar;
byte voltageBatch[64] = {};
byte risingArray[64] = { 0 };
byte fallingArray[64] = { 0 };
volatile int buttonFlag = 0;
volatile int stageTriggerFlag = 0;
int lastButtonState = 1; // indicator of current scanning state

void blink_func(int on_time, int off_time);
void ISR_button();
void ISR_trigger();

// the setup function runs once when you press reset or power the board
void setup() {
	// initialize Teensy-to-RS232-to-StgDriver:
	Serial1.begin(115200);

	// initialize USB-to-PC:  
	Serial.begin(115200);

	// initialize interrupt function:
	attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), ISR_button, CHANGE);
  attachInterrupt(digitalPinToInterrupt(DIGITAL_TRIG), ISR_trigger, CHANGE);

	// highest priority tointerrupt - to 'interrupt' "delay"-fuction:
	NVIC_SET_PRIORITY(IRQ_PORTE, 2);

	pinMode(LED_PIN, OUTPUT); // initialize LED on Teensy

	analogReadResolution(16);
	analogReadAveraging(2);

	// To show that 'setup' is done:
	blink_func(1000, 0);

  memset(risingArray, 255, sizeof(risingArray));
  memset(fallingArray, 254, sizeof(fallingArray));
}

// the loop function runs over and over again until power down or reset
void loop() {
	if (buttonFlag == 1) {
		blink_func(500, 100);
		lastButtonState = 0;
		buttonFlag = 0;
	}

	if (lastButtonState == 0) { // if scanning:
    //Serial.println("Scanning");
    
		while (1) {
      //Serial.println(digitalRead(DIGITAL_TRIG));
      voltageBatch[64] = {};
			for (counter = 0; counter <= 15; counter++){
				refIntV = analogRead(REFINT_PIN); // read stage position
  			voltageBatch[0 + counter * 4] = highByte(refIntV); // 
  			voltageBatch[1 + counter * 4] = lowByte(refIntV); // 
  			intV = analogRead(INT_PIN); // read signal from detector
  			voltageBatch[2 + counter * 4] = highByte(intV); // 
  			voltageBatch[3 + counter * 4] = lowByte(intV); // 
			}
      Serial.write(voltageBatch,sizeof(voltageBatch));
			if (buttonFlag == 1) {
				blink_func(500, 100);
				blink_func(500, 100);
				lastButtonState = 1;
				buttonFlag = 0;
				break;
			}
      if (stageTriggerFlag == 1){
        Serial.write(risingArray,sizeof(risingArray));
        stageTriggerFlag = 0;
      }
      else if (stageTriggerFlag == 2){
        Serial.write(fallingArray, sizeof(fallingArray));
        stageTriggerFlag = 0;
      }
		}
	}
	else if ((Serial.available() > 0) & (lastButtonState == 1)) {
		outgoingChar = Serial.read();
		Serial1.print(outgoingChar);
		blink_func(50, 50); // necessary for proper reading/writing
		while (Serial1.available() > 0) {
			incomingChar = Serial1.read();
			Serial.print(incomingChar);
		}
	}
}

void blink_func(int on_time, int off_time) {
	digitalWrite(LED_PIN, HIGH);
	delay(on_time);
	digitalWrite(LED_PIN, LOW);
	delay(off_time);
}

// interruption function:
void ISR_button() { 
	buttonFlag = 1;
}

// interruption to trigger change
void ISR_trigger(){
  if (digitalRead(DIGITAL_TRIG) == HIGH){
    stageTriggerFlag = 1;
  }
  else {
    stageTriggerFlag = 2;
  }
}



