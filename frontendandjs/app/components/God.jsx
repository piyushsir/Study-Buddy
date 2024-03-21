import React from "react";
import { sourceCodePro } from "../styles/fonts";

const God = ({
  handleSubmit,
}) => {
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };
  return (
    <>
       <button>press</button> 
    </>
        )
    }

export default God;