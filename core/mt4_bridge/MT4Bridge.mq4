
// MT4 Python Bridge DLL
// Compile with: mql4 compiler in MT4
#property copyright "Rafał Wiśniewski"
#property link      "github.com/RafalWisniewski"
#property strict

#import "kernel32.dll"
   int CreateFileW(string,int,int,int,int,int,int);
   int WriteFile(int,uchar&[],int,int&[],int);
   int CloseHandle(int);
#import

// Send optimization request to Python
void SendToPython(string symbol, int timeframe, double params[]) {
   // Implementation pending
}
