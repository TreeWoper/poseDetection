const express = require("express");
const { exec } = require("child_process");
const cors = require("cors");

const app = express();
const PORT = 5000;

app.use(cors()); // allow requests from frontend

app.get("/run-python", (req, res) => {
  // run your python script
  exec("python ../../poseFinder.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return res.status(500).send("Error running Python script");
    }
    res.send(`Python script output:\n${stdout}`);
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
