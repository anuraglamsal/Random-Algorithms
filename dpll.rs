// implementing the dpll algorithm for SAT solving (https://en.wikipedia.org/wiki/DPLL_algorithm)
// in Rust. symbolic algebra library used for the evaluation: https://github.com/p-e-w/savage
// I'm like 90% sure that this is correct. 

use std::collections::HashMap;

use savage_core::{expression::Expression};

fn main() {
    let exp = "!x && (!x || y) && (!x || z || y) && (p || m)".parse::<Expression>().unwrap(); // true
    let exp_2 = "!x && x".parse::<Expression>().unwrap(); // false
    let exp_3 = "(a || !b || d) && (a || !b || e) && (!b || !d || !e) && (a || b || c || d) && (a || b || c || !d) && (a || b || !c || e) && (a || b || !c || !e)".parse::<Expression>().unwrap(); // true
    let exp_4 = "(x || y || z) && (x || y || !z) && (x || !y || z) && (x || !y || !z) && (!x || y || z) && (!x || y || !z) && (!x || !y || z) && (!x || !y || !z)".parse::<Expression>().unwrap(); // false

    println!("{:?}", dpll(exp)); // gives true
    println!("{:?}", dpll(exp_2)); // gives false
    println!("{:?}", dpll(exp_3)); // gives true
    println!("{:?}", dpll(exp_4)); // gives false
}

// note: implementing the "Copy" trait for a type provides
// implicit copying. If that is not implemented, you need to use
// "clone" on the variables of that type if you don't want to 
// transfer ownership (when sending them to a function for example.)
fn dpll(exp: Expression) -> bool {
    // Stores the variables in our expression with pertinent info about them.

    // key = variable name. value = tuple of size 2 of Option<bool>.
    
    // value.0 = None if not a unit clause, Some(False) if the unit
    // clause is '!variable' and Some(True) if the unit clause is
    // 'variable'.
    
    // value.1 = None if not a pure literal, Some(False) if the pure
    // literal is in the form '!variable' and Some(True) if the pure 
    // literal is in the form 'variable.'
    let mut vars = HashMap::new();

    let exp_clone = exp.clone();
    extract_variables(exp_clone, &mut vars);

    // println!("{:?}", vars); // run this to see the beautiful hashtable.

    let mut context = HashMap::new(); // you need to store the variables to evaluate along with 
                                      // their values in a hashmap. this is what the sym alg library
                                      // dictates. 
    for (key, value) in vars.into_iter() {
        if value.0 != None {
            context.insert(key, Expression::Boolean(value.0.unwrap()));
        }
        else if value.1 != None {
            context.insert(key, Expression::Boolean(value.1.unwrap()));
        }
    }

    let res_exp = exp.evaluate(&context).unwrap(); // the evaluation

    if let Expression::Boolean(value) = res_exp { // base case
        return value;
    }
 
    // extract out a variable from the resulting expression after evaluation.
    let mut extract_var: String = String::new(); 
    for _exp in res_exp.parts() { 
        if let Expression::Variable(identifier) = _exp {
            extract_var = identifier;
        }
    }

    let mut cont_1 = HashMap::new();
    let mut cont_2 = HashMap::new();

    cont_1.insert(extract_var.clone(), Expression::Boolean(true));
    cont_2.insert(extract_var, Expression::Boolean(false));

    // short-circuit evaluation: https://github.com/anuraglamsal/playground/blob/main/rust_short_circuit.rs
    return dpll(res_exp.evaluate(&cont_1).unwrap()) || dpll(res_exp.evaluate(&cont_2).unwrap()); 
}

// You have a binary tree of expressions.

// Note: 'a' and 'b' are 'Box'es of expressions, therefore, we are dereferencing using '*' to
// get the expression. Btw, when I say expression, I am talking about the library's enum called
// 'Expression'. An 'Expression' can be 'Variable', 'And', 'Not', 'Or, etc. 

fn extract_variables(exp: Expression, vars: &mut HashMap<String, (Option<bool>, Option<bool>)>) {
    match exp {
        Expression::And(a, b) => { // if the parent node is And, then the child, if a variable, is a unit clause.
            let a_clone = *a.clone();
            let b_clone = *b.clone();
            handle_and_vars(*a, check_variable(a_clone), vars);
            handle_and_vars(*b, check_variable(b_clone), vars);
        },
        Expression::Or(a, b) => { // if the parent node is Or, then the child, if a variable, is a candidate for 
                                  // being a pure literal at least.
            let a_clone = *a.clone();
            let b_clone = *b.clone();
            handle_or_vars(*a, check_variable(a_clone), vars);
            handle_or_vars(*b, check_variable(b_clone), vars);
        },
        _ => () // ignore others as the expressions can only be And, Or, Not(Var) and Var. 
                // Not(Var) and Var are checked in another function below.
    }
}

// populates the hashtable for variables whose parent is an And. 
fn handle_and_vars(exp: Expression, res: (String, Option<bool>), vars: &mut HashMap<String, (Option<bool>, Option<bool>)>) {
    if res.1 != None { // reading and understanding is better than explaining tbh. 
        if let Some(value) = vars.get_mut(&res.0) {
            value.0 = res.1;
        }
        else {
            vars.insert(res.0, (res.1, res.1));
        }
    }
    else {
        extract_variables(exp, vars); // continue going down the tree if the child isn't a 'Var' or 'Not(Var)'. 
    }
}

// populates the hashtable for variables whose parent is an 'Or'.
fn handle_or_vars(exp: Expression, res: (String, Option<bool>), vars: &mut HashMap<String, (Option<bool>, Option<bool>)>) {
    if res.1 != None { // reading and understanding is better than explaining tbh. 
        if let Some(value) = vars.get_mut(&res.0) {
            if value.1 != None {
                if value.1 != res.1 {
                    value.1 = None;
                }
            }
        }
        else {
            vars.insert(res.0, (None, res.1)); 
        }
    }
    else {
        extract_variables(exp, vars); // continue going down the tree if the child isn't a 'Var' or 'Not(Var)'. 
    }
}

// checks whether the expression inside an 'And' or an 'Or' is a 'Var' or 'Not(Var)'.
fn check_variable(exp: Expression) -> (String, Option<bool>) { 
    match exp { 
        Expression::Variable(a) => {
            return (a.to_owned(), Some(true));
        },
        Expression::Not(a) => {
            match *a { // the content of "Not" is a box too. 
                Expression::Variable(a) => {
                    return (a.to_owned(), Some(false));
                },
                _ => return ("".to_owned(), None)
            }
        },
        _ => {
            return ("".to_owned(), None);
        }
    }
}
