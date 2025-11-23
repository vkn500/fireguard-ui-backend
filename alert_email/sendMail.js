// sendMail.js
const nodemailer = require("nodemailer");

async function sendEmail(status, severity, timestamp) {
  let transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: "infomotivefact@gmail.com",
      pass: "qlnl dqub rlfx tpsa"  
    }
  });

  let message = `
🔥 FIRE / SMOKE ALERT

Status   : ${status}
Severity : ${severity}
Time     : ${timestamp}

Please take necessary action.
`;

  let info = await transporter.sendMail({
    from: '"Fire Alert System" <YOUR_EMAIL@gmail.com>',
    to: "vkngaming8085@gmail.com",
    subject: `🔥 Fire Alert | Status: ${status} | Severity: ${severity}`,
    text: message
  });

  console.log("📧 Email sent:", info.response);
}

// Run email function only when called from Python
sendEmail(process.argv[2], process.argv[3], process.argv[4]);
