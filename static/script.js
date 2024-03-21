//if(typeof window!="undefined"){
console.log(1);
function readandwrite(){
    const fs = require('fs');
    const filePath = 'C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/characteristic.txt';
    fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
    console.error('Error reading file:', err);
    return;
    }
    console.log(data);
    console.log(typeof(data));
    document.getElementsByTagName("textarea").value="data";
    //cd.value="abcd"
   });

}
    
function sayhello()
{
    document.getElementById("123").value = "bacd"
}
//}