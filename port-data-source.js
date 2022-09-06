const katex = require('katex')
module.exports = {
    /**
       * @param { unknown } fromElm
       * @returns { Promise<unknown> }
       */
    parse_katex: async function (formula) {
        console.log({ formula })
        const val = katex.renderToString(formula, {
            displayMode: true,
            throwOnError: true
        })
        return val
    },

    /**
       * @param { unknown } fromElm
       * @returns { Promise<unknown> }
       */
    environmentVariable: async function (name) {
        const result = process.env[name]
        if (result) {
            return result
        } else {
            throw `No environment variable called ${kleur
                .yellow()
                .underline(name)}\n\nAvailable:\n\n${Object.keys(process.env).join(
                    "\n"
                )}`
        }
    }
}