"use client";
import React, { useState, useEffect } from "react";
import PromptBox from "../components/PromptBox";
import ResultStreaming from "../components/ResultStreaming";
import Title from "../components/Title";
import TwoColumnLayout from "app/components/TwoColumnLayout";


const Streaming = () => {
  const [prompt, setPrompt] = useState("");
  const [tmp,settmp] =useState("")
  const [error, setError] = useState(null);
  const [data, setData] = useState("");
  const [source, setSource] = useState(null);
  const [mess1,setmess1]=useState("interview is processing")
  const[mess2,setmess2] =useState("Done Interviewing Type and Press 'enter' ")
  useEffect(()=>{
    fetch("http://127.0.0.1:8080/api/home").then(
     response => response.json()
    ).then(
     data=>{
       settmp(data.data)
     }
    )
 },[])

  const processToken = (token) => {
    return token.replace(/\\n/g, "\n").replace(/\"/g, "");
  };
  
  const handlePromptChange = () => {
    setPrompt(tmp);
  };

  const handleSubmit = async () => {
    try {
      console.log(`sending ${prompt}`);
      await fetch("/api/streaming", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: prompt }),
      });
      // close existing sources
      if (source) {
        source.close();
      }
      // create new eventsource

      const newSource = new EventSource("/api/streaming");

      setSource(newSource);

      newSource.addEventListener("newToken", (event) => {
        const token = processToken(event.data);
        
        setData((prevData) => prevData + token);
      });

      newSource.addEventListener("end", () => {
        newSource.close();
      });
    } catch (err) {
      console.error(err);
      setError(error);
    }
  };

  // Clean up the EventSource on component unmount
  useEffect(() => {
    // stuff is gonna happen
    return () => {
      if (source) {
        source.close();
      }
    };
  }, [source]);
  return (
    <>
      <Title emoji="💭" headingText="Interview" />
      <TwoColumnLayout
        leftChildren={
          <>
            <ResultStreaming data={data} />
            <div></div>
            <PromptBox
              prompt={prompt}
              handlePromptChange={handlePromptChange}
              handleSubmit={handleSubmit}
              placeHolderText={tmp==""?mess1:mess2}
              error={error}
              pngFile="pdf"
            />
          </>
          
        }
        rightChildren={
          <>
           
          </>
        }
      />
    </>
  );
};

export default Streaming;
